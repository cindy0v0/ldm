import inspect
import os
import contextlib
from typing import Callable, List, Optional, Union, Tuple, Dict

import torch
import torch.nn as nn
from diffusers import (
    VQModel,
    UNet2DModel,
    AutoencoderKL
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from diffusers.utils.torch_utils import randn_tensor


class LatentDiffusionPipelineBase(DiffusionPipeline):

    @torch.no_grad()
    def prepare_latents(self, batch, noise_timesteps=350):
        with self.autocast_ctx:
            vae = self.vae
            noise_scheduler = self.scheduler
            
            clean_images = batch["input"].to(vae.device)
            posterior = vae.encode(clean_images)
            if hasattr(posterior, "latent_dist"):
                latents = posterior.latent_dist.sample()
            else:
                latents = posterior.latents

            vae_shift = vae.config.shift_factor if hasattr(
                vae.config, "shift_factor") else 0.0
            latents = (latents - vae_shift) * vae.config.scaling_factor
                
            # Sample noise that we'll add to the images
            noise = torch.randn(latents.shape, 
                                device=latents.device)
            bsz = latents.shape[0]
            timesteps = torch.LongTensor(
                [noise_timesteps] * bsz).to(vae.device)

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)
        return noisy_latents, latents

    @torch.no_grad()
    def decode_latents(self, latents):
        with self.autocast_ctx:
            vae_shift = self.vae.config.shift_factor if hasattr(
                self.vae.config, "shift_factor") else 0.0
                
            latents = latents / self.vae.config.scaling_factor + vae_shift
            # latents = latents.to(self.vae.dtype)
            image = self.vae.decode(latents, return_dict=False)[0]
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.cpu().permute(0, 2, 3, 1).float()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # def prepare_latents_(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    #     shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
    #     if isinstance(generator, list) and len(generator) != batch_size:
    #         raise ValueError(
    #             f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
    #             f" size of {batch_size}. Make sure the batch size matches the length of the generators."
    #         )

    #     if latents is None:
    #         latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    #     else:
    #         if latents.shape != shape:
    #             raise ValueError(
    #                 f"Unexpected latents shape, got {latents.shape}, expected {shape}"
    #             )
    #         latents = latents.to(device)

    #     # scale the initial noise by the standard deviation required by the scheduler
    #     latents = latents * self.scheduler.init_noise_sigma
    #     return latents


class UncondLatentDiffusionPipeline(LatentDiffusionPipelineBase):
    def __init__(
            self,
            vae: Union[VQModel, AutoencoderKL],
            scheduler: Union[
                DDIMScheduler,
                DDPMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                LMSDiscreteScheduler,
                PNDMScheduler,
            ],
            unet: UNet2DModel,
            autocast_dtype: torch.dtype = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)

        self.autocast_ctx = torch.autocast(
            "cuda", dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
        
    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,  # default to generate a single image
            height: Optional[int] = None,
            width: Optional[int] = None,
            generator: Optional[Union[torch.Generator,
                                      List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            batch: Optional[dict] = None,
            num_inference_steps: Optional[int] = 50,
            noise_timesteps: int = 350,
            output_type: Optional[str] = "numpy",
            return_dict: bool = False,
            eta: Optional[float] = 0.0,
            **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput, Dict[str, torch.Tensor]]:

        self.scheduler.set_timesteps(num_inference_steps)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        
        with self.autocast_ctx:
            latents, clean_latents = self.prepare_latents(
                batch, noise_timesteps=noise_timesteps)

            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            for t in self.progress_bar(self.scheduler.timesteps):
                latents = self.scheduler.scale_model_input(latents, t)

                noise_pred = self.unet(
                    latents,
                    t,
                ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

            image = self.decode_latents(latents)
            clean_image = self.decode_latents(clean_latents)

        if output_type == "pil":
            image = self.numpy_to_pil(image.cpu().numpy())

        if return_dict:
            return {'images': image,
                    'clean_images': clean_image,
                    'latents': latents,
                    'clean_latents': clean_latents}

        return ImagePipelineOutput(images=image)
