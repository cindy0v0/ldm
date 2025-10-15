import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
from PIL import Image

import torch
import torch.fft as fft
import torch.nn.functional as F
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D, Upsample2D, Downsample2D
from diffusers.models.unets.unet_2d_blocks import AttnUpBlock2D, UpBlock2D

import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
import lpips

import diffusers
from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL, VQModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from src.pipeline import *
from src.data import PathologyTrain
from src.standardizer import ChannelStandardize

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# Change the info of your pretrained VAE model here
VAE_PRETRAINED_PATH = "CompVis/ldm-celebahq-256"
VAE_KWARGS = {"subfolder": "vqvae"}


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--vae_config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-256",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="amount of jitter to set for torch colorjitter transform"
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--save_images_epochs", type=int, default=1,
                        help="How often to save images during training.")
    parser.add_argument("--image_logging_steps", type=int, default=10000,)
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0,
                        help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4,
                        help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999,
                        help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="ldm",
        help="The name of the project for logging to TensorBoard or Weights and Biases.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="ldm-finetune",
        help="The name of the experiment for logging to TensorBoard or Weights and Biases.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16", "fp32"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample", "v_prediction"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--ddpm_noise_timesteps", type=int, default=350,)
    parser.add_argument("--gamma", type=float, default=5.0,)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--acc_seed",
        type=int,
        default=42,
        help="A seed to reproduce the training. If not set, the seed will be random.",
    )
    parser.add_argument(
        "--train_data_files",
        type=str,
        default=None,
        help=(
            "The files of the training data. The files must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--swap_conv3",
        action="store_true"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def zero_init_out(unet):
    if hasattr(unet, "conv_out"):
        nn.init.zeros_(unet.conv_out.weight)
        if unet.conv_out.bias is not None:
            nn.init.zeros_(unet.conv_out.bias)
    # (Optional) damp last convs in each up block
    for idx, ub in enumerate(unet.up_blocks):
        if hasattr(ub, "resnets") and len(ub.resnets):
            last = ub.resnets[-1]
            for name, m in last.named_modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


def set_bilinear_upblock(model):
    def aa_bilinear_upsample_forward(self, hidden_states, *args, **kwargs):
        # hidden_states: (B, C, H, W)
        # upsample by 2 with antialiasing, then 3x3 conv
        x = F.interpolate(
            hidden_states, scale_factor=2, mode="bilinear",
            align_corners=False, antialias=True
        )
        if self.conv is not None:
            x = self.conv(x)
        return x

    for b, _ in enumerate(model.up_blocks):
        block = model.up_blocks[b]
        upsamplers = getattr(block, "upsamplers", [])
        if not upsamplers:
            continue
        for u, _ in enumerate(upsamplers):
            up = upsamplers[u]
            up.forward = aa_bilinear_upsample_forward.__get__(up, Upsample2D)
            print("set bilinear upsample for ", up)

def swap_first_resnet_conv1_to_1x1(unet):
    for u, ub in enumerate(unet.up_blocks):
        # only the last up block
        if u == 3:
            if not hasattr(ub, "resnets") or len(ub.resnets) == 0:
                continue
            rb0 = ub.resnets[0]                    # first ResNet after concat
            if isinstance(rb0, ResnetBlock2D):
                old = rb0.conv1
                rb0.conv1 = nn.Conv2d(
                    in_channels=old.in_channels,
                    out_channels=old.out_channels,
                    kernel_size=1, bias=(old.bias is not None)
                )
                nn.init.kaiming_normal_(rb0.conv1.weight, nonlinearity="linear")
                if rb0.conv1.bias is not None: nn.init.zeros_(rb0.conv1.bias)
            print("set final upblock conv3 to be conv1")

def depthwise_blur3x3(x):
    k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=x.dtype, device=x.device)/16
    w = k.expand(x.size(1),1,3,3)
    return F.conv2d(x, w, padding=1, groups=x.size(1))

def rms_branch(x, eps=1e-6):
    # per-sample RMS over (C,H,W)
    return x.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().clamp_min(eps)

def wrap_upblock_preblur_equalize(block):
    orig_forward = block.forward
    def forward_wrapped(hidden_states, res_hidden_states_tuple, temb=None, **kw):
        # pre-blur and equalize *each* skip used in this block
        # res_hidden_states_tuple is a tuple of tensors, one per resnet in the block
        new_res = []
        for r in res_hidden_states_tuple:
            r = depthwise_blur3x3(r)
            # equalize RMS vs current up-stream to prevent skip dominance
            g = rms_branch(hidden_states) / rms_branch(r)
            g = g.clamp(0.5, 2.0)  # keep stable
            r = r * g
            new_res.append(r)
        return orig_forward(hidden_states, tuple(new_res), temb=temb, **kw)
    block.forward = forward_wrapped

class Blur(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        # fixed depthwise gaussian-ish kernel
        w = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32)
        w = (w / w.sum()).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        self.register_buffer("weight", w)
        self.pad = nn.ReflectionPad2d((1,1,1,1))
    def forward(self, x):
        C = x.size(1)
        weight = self.weight.expand(C, 1, 3, 3)
        return nn.functional.conv2d(self.pad(x), weight, groups=C)

def add_blur_around_resampling(unet):
    # after every upsample
    for b in unet.up_blocks:
        upsamplers = b.upsamplers
        if not upsamplers:
            continue
        for i, m in enumerate(upsamplers):
            upsamplers[i] = nn.Sequential(m, Blur(3))
    # before every downsample
    for b in unet.down_blocks:
        downsamplers = getattr(b, "downsamplers", [])
        if not downsamplers:
            # occassionally there is no up/downsampler but attr exist with value None
            continue
        for i, m in enumerate(downsamplers):
            downsamplers[i] = nn.Sequential(Blur(3), m)

def block_fft_score(model):
    def checker_parity_score(input):
        """
        x: [B,C,H,W], returns a scalar measuring 2x2 checker energy.
        Works even for 4x4. High = stronger checkerboard.
        """
        x = input.to(torch.float32).detach()
        B, C, H, W = x.shape
        # +1/-1 parity mask: (-1)^(i+j)
        ii = torch.arange(H, device=x.device).view(H, 1)
        jj = torch.arange(W, device=x.device).view(1, W)
        parity = ((ii + jj) % 2) * 2 - 1  # { -1, +1 }
        parity = parity.view(1,1,H,W).to(x.dtype)
        y = (x * parity).mean(dim=(1,2,3))  # per-sample
        return y.abs().mean().item()

    def grid_score_fft(input):
        """
        x: [B,C,H,W]. Use FFT only when H,W >= 16.
        Measures high-freq corner energy (aliasing/gridty stuff).
        """
        x = input.to(torch.float32).detach()
        B, C, H, W = x.shape
        if H < 16 or W < 16:
            return float('nan')
        g = x.mean(1, keepdim=True)
        F = fft.fftshift(fft.fft2(g, norm="ortho"))
        mag = (F.real**2 + F.imag**2).sqrt()
        # take a small border band (corners) as high-freq proxy
        k = max(2, min(H, W) // 16)  # band thickness
        top    = mag[..., :k, :].mean(dim=(-2,-1)) # [B,1]
        bottom = mag[..., -k:, :].mean(dim=(-2,-1)) # [B,1]
        left   = mag[..., :, :k].mean(dim=(-2,-1)) # [B,1]
        right  = mag[..., :, -k:].mean(dim=(-2,-1)) # [B,1]
        corner_energy = torch.cat([top, bottom, left, right], dim=-1).mean()
        return float(corner_energy)

    def res_hook(module, input, output):
        hidden_states = input[0]
        res_hidden_states_tuple = input[1]
        i_fft = grid_score_fft(hidden_states)
        i_parity = checker_parity_score(hidden_states)
        for attn_idx in range(len(res_hidden_states_tuple)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            o_fft = grid_score_fft(hidden_states)
            o_parity = checker_parity_score(hidden_states)
            print(f"{module.__class__.__name__.lower()} \
                {hidden_states.shape[2]} {output.shape[2]} \
                    {i_fft:5f} {i_parity:5f} \
                        {o_fft:5f} {o_parity:5f} \
                            {(o_fft - i_fft):5f}, {(o_parity - i_parity):5f} \n")
            i_fft = o_fft
            i_parity = o_parity
    
    def hook(module, input, output):
        # hook receives a tuple of inputs
        i_fft = grid_score_fft(input[0])
        o_fft = grid_score_fft(output)
        i_checker = checker_parity_score(input[0])
        o_checker = checker_parity_score(output)
        
        print(f"{module.__class__.__name__.lower()} \
            {input[0].shape[2]} {output.shape[2]} \
              {i_fft:5f} {i_checker:5f} | {o_fft:5f} {o_checker:5f} \
                  {(o_fft - i_fft):5f}, {(o_checker - i_checker):5f} ")
        
    # model.up_blocks[0].attentions[0].register_forward_hook(hook)
    for up, _ in enumerate(model.up_blocks):
        if up == 3:
            block = model.up_blocks[up]
            # if block.upsamplers:
            #     block.upsamplers[0].register_forward_hook(hook)
            # block.resnets[0].register_forward_hook(hook)
            block.register_forward_hook(res_hook)

# check grad stability
def nan_hook(name):
    def _hook(module, grad_in, grad_out):
        for gi in grad_in:
            if gi is not None and not torch.isfinite(gi).all():
                print(f"NaN in grad_in of {name}")
                raise RuntimeError("NaN detected")
        for go in grad_out:
            if go is not None and not torch.isfinite(go).all():
                print(f"NaN in grad_out of {name}")
                raise RuntimeError("NaN detected")
    return _hook

def checker_loss(img, w_parity=1e-5, w_fft=1e-6):
    # img: [B,3,H,W] decoded prediction (clamp to [-1,1] or [0,1] consistently)
    g = img.mean(1, keepdim=True)

    # 2×2 parity term (works at any size)
    H,W = g.shape[-2:]
    ii = torch.arange(H, device=g.device).view(H,1)
    jj = torch.arange(W, device=g.device).view(1,W)
    parity = (((ii+jj) % 2)*2 - 1).view(1,1,H,W).to(g.dtype)
    parity_energy = (g*parity).mean().abs()

    # notch the 2-px & 4-px lattice in FFT
    F = fft.fftshift(fft.fft2(g, norm="ortho"))
    mag = (F.real**2 + F.imag**2).sqrt()
    k = max(2, min(H,W)//16)
    corners = torch.stack([
        mag[..., :k, :k], mag[..., :k, -k:],
        mag[..., -k:, :k], mag[..., -k:, -k:]
    ]).mean()
    return w_parity*parity_energy + w_fft*corners

def tv_loss(x̂):
    tv_loss = ((x̂[:, :, :, 1:] - x̂[:, :, :, :-1]).abs().mean() +
                (x̂[:, :, 1:, :] - x̂[:, :, :-1, :]).abs().mean())
    return 1e-5 * tv_loss

# def fft_band_l1(x,y,lo=0.2,hi=0.6,eps=1e-8):
#     X = torch.fft.rfft2(x, norm='ortho'); Y = torch.fft.rfft2(y, norm='ortho')
#     H,W = x.shape[-2:]
#     yy, xx = torch.meshgrid(torch.linspace(0,1,H,device=x.device),
#                             torch.linspace(0,1,W//2+1,device=x.device), indexing='ij')
#     r = torch.sqrt((xx-0.0)**2 + (yy-0.0)**2)  # radial freq
#     M = ((r>=lo) & (r<=hi)).float()
#     gt = torch.log(torch.abs(X)+eps)
#     rec = torch.log(torch.abs(Y)+eps)
#     return (M*(gt-rec).abs()).mean(), gt, rec

def fft_band_l1_img(x, y, lo=0.2, hi=0.6, eps=1e-8):
    # x,y: [-1,1] images (B,C,H,W)
    B,C,H,W = x.shape
    win_h = torch.hann_window(H, device=x.device).unsqueeze(1)
    win_w = torch.hann_window(W, device=x.device).unsqueeze(0)
    win = (win_h @ win_w).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    X = torch.fft.rfft2(x*win, norm="ortho")
    Y = torch.fft.rfft2(y*win, norm="ortho")

    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=x.device),
        torch.linspace(0, 1, W//2+1, device=x.device),
        indexing="ij"
    )
    r = torch.sqrt(xx**2 + yy**2)
    M = ((r>=lo) & (r<=hi)).float()  # (H, W//2+1)

    magX = torch.log(torch.abs(X)+eps)
    magY = torch.log(torch.abs(Y)+eps)
    rel  = (magX - magY).abs() / (magY.abs() + 1.0)  # relative log-mag

    return (rel * M).mean()

def topk_grad_modules(model, k=5):
    stats = []
    for n,p in model.named_parameters():
        if p.grad is None: continue
        g = p.grad.detach()
        stats.append((n, float(g.norm(2))))
    norm = math.sqrt(sum([s[1]**2 for s in stats]))
    return sorted(stats, key=lambda x: x[1], reverse=True)[:k], norm



@torch.no_grad()
def agc_(params, clip=0.5, eps=1e-6):
    for p in params:
        if p.grad is None: continue
        pnorm = p.norm().clamp_min(eps)
        gnorm = p.grad.norm()
        if gnorm > clip * pnorm:
            p.grad.mul_((clip * pnorm) / (gnorm + 1e-6))  

def clip_by_value_modules(model, modules, clip_val=1.0):
    for n, p in model.named_parameters():
        if p.grad is None: continue
        if any(tag in n for tag in modules):
            p.grad.data.clamp_(-clip_val, clip_val)


def finite_or_none(x):
    return x is None or torch.isfinite(x).all()

def loss_is_finite(loss):
    return torch.isfinite(loss)

def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)

    # a big number for high resolution or big dataset
    # todo: log config for reproducibility
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError(
                "Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Set the random seed manually for reproducibility.
    if args.acc_seed is not None:
        set_seed(args.acc_seed)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(
                        os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    try:
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    except:
                        # skips in adapter
                        continue

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load pretrained VAE model
    if args.vae_config == "CompVis/ldm-celebahq-256":
        vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_config, subfolder="vae")

    # Freeze the VAE model
    vae.requires_grad_(False)
    vae.eval()

    vae_shift = vae.config.shift_factor if hasattr(
        vae.config, "shift_factor") else 0.0
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # Initialize the model
    if args.model_config_name_or_path is None:
        model = UNet2DModel(sample_size=args.resolution // vae_scale_factor,
                            in_channels=vae.config.latent_channels, out_channels=vae.config.latent_channels)
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)
    if args.swap_conv3:
        for ub in model.up_blocks:
            if isinstance(ub, (UpBlock2D, AttnUpBlock2D)):
                wrap_upblock_preblur_equalize(ub)
        swap_first_resnet_conv1_to_1x1(model)
        set_bilinear_upblock(model)
    
    zero_init_out(model)
    add_blur_around_resampling(model)
    
    # Attach to trouble spots
    if hasattr(model, "conv_out"):
        model.conv_out.register_full_backward_hook(nan_hook("conv_out"))
    # time embedding path (names vary)
    for n, m in model.named_modules():
        if "time" in n and isinstance(m, nn.Linear):
            m.register_full_backward_hook(nan_hook(n))
    
    
    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(
        inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        print("defaulted to epsilon prediction")
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
    noise_scheduler.clip_sample = False
    
    # Initialize the optimizer
    decay, no_decay = set(), set()
    for n,p in model.named_parameters():
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n or "bn" in n or "gn" in n:
            no_decay.add(n)
        else:
            decay.add(n)
    optimizer = torch.optim.AdamW(
        [
            {"params":[p for n,p in model.named_parameters() if n in decay],    "weight_decay":1e-6},
            {"params":[p for n,p in model.named_parameters() if n in no_decay], "weight_decay":0.0},
        ],
        lr=1e-4, betas=(0.9,0.999), eps=1e-6
    )
    
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )
    
    # dec_params = list(model.up_blocks[-1].resnets.parameters())
    # model_rest = list(model.up_blocks[:-1].parameters()) + list(model.mid_block.parameters()) + list(model.down_blocks.parameters()) + \
    #                 list(model.conv_out.parameters()) + list(model.conv_norm_out.parameters()) + \
    #                     list(model.up_blocks[0].attentions.parameters()) + list(model.up_blocks[0].attentions.parameters()) + \
    #                         list(model.time_embedding.parameters()) + list(model.time_proj.parameters()) + \
    #                             list(model.conv_in.parameters())
    # if hasattr(model, "class_embedding") and model.class_embedding:
    #     model_rest += list(model.class_embedding.parameters()) 
    # optimizer = torch.optim.AdamW([
    #     {"params": dec_params, "lr": 1e-5, "weight_decay": 0.0},   # encoder-side bottleneck
    #     {"params": model_rest, "lr": 1e-4,  "weight_decay": 1e-6}, 
    # ], betas=(0.9, 0.999))

    # paths = "/projects/ovcare/users/cindy_shi/ldm/uncond-image-generation-ldm/ddpm-model-256/image_paths"
    # with open(paths, "r") as f:
    #     paths = f.read().splitlines()
    # Load the dataset
    dataset = PathologyTrain(
        size=256, n_samples=args.n_samples, jitter=args.jitter)
    train_dataloader = torch.utils.data.DataLoader(dataset, 
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    pipeline = UncondLatentDiffusionPipeline(
        vae=vae.to("cuda"),
        unet=model.to("cuda"),
        scheduler=noise_scheduler,
    )
    
    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )
    
    # in_adapter = nn.Conv2d(vae.config.latent_channels, vae.config.latent_channels, 1)
    # nn.init.dirac_(in_adapter.weight); nn.init.zeros_(in_adapter.bias)

    lpips_loss = lpips.LPIPS(net='vgg').requires_grad_(False).to(accelerator.device)
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    vae = vae.to(accelerator.device) 

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(
            project_name=args.project_name,
            config=args,
            init_kwargs={"wandb": {"name": args.experiment_name}}
        )

    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    grad_norm = torch.zeros(1)
    
    test_batch = {'input': torch.stack([dataset[i]['input'] for i in range(
            min(5, len(dataset)))]).to(weight_dtype).to(accelerator.device)}

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[-1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # todo: code for debugging sai checkerboard
    # block_fft_score(model)

    # Train!
    _checker_loss = torch.zeros(1)
    _lpips_loss = torch.zeros(1)
    _tv_loss = torch.zeros(1)
    _hf_loss = torch.zeros(1)
    _best_lpips_loss = torch.tensor(1e9)
    preclip_norm = 0
    modules = ()
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with torch.autocast("cuda", dtype=weight_dtype, enabled=(args.mixed_precision != "no")):
                clean_images = batch["input"]
                posterior = vae.encode(clean_images)
            if hasattr(posterior, "latent_dist"):
                clean_latents = posterior.latent_dist.sample()
            else:
                clean_latents = posterior.latents
            latents = (clean_latents - vae_shift) * vae.config.scaling_factor
            
            # std channels
            # if vae.config.latent_channels == mean.numel():
            #     latents = chan_std(clean_latents)
            # latents = in_adapter(latents)
            
            # Sample noise that we'll add to the images
            noise = torch.randn(
                latents.shape, dtype=weight_dtype, device=latents.device)
            bsz = latents.shape[0]  # batch size
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_latents, timesteps).sample

                # loss computation
                alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (
                            clean_images.shape[0], 1, 1, 1)
                    ).float().clamp_(1e-6, 1-1e-6)
                if args.prediction_type == "epsilon":
                    # this could have different weights!
                    bsz = model_output.shape[0]
                    snr_weights = alpha_t / (1 - alpha_t)
                    snr_weights = torch.min(torch.tensor([1.]).to(snr_weights.device), torch.tensor(
                        [float(args.gamma)]).to(snr_weights.device)/(snr_weights + 1e-9))
                    snr_weights[snr_weights == 0] = 1.0
                    loss = snr_weights * \
                        F.mse_loss(model_output.float(),
                                    noise.float(), reduction="none")
                    loss = loss.mean()
                    
                elif args.prediction_type == "v_prediction":
                    # v   = √α_t * ε − √(1−α_t) * x0
                    # let the scheduler convert v_pred→ε_pred/x0_pred internally
                    # v prediction training objective from https://arxiv.org/abs/2202.09778
                    sigma_t = torch.sqrt(1 - alpha_t)
                    target = torch.sqrt(alpha_t) * noise - sigma_t * latents
                    loss = F.mse_loss(model_output.float(),
                                      target.float(), reduction="none")
                    loss = loss.mean(dim=[1, 2, 3])
                    snr_weights = alpha_t / (1 - alpha_t)
                    snr_weights = torch.min(torch.tensor([1.]).to(snr_weights.device), torch.tensor(
                        [float(args.gamma)]).to(snr_weights.device)/(snr_weights + 1e-9))
                    snr_weights[snr_weights == 0] = 1.0
                    loss = snr_weights * loss
                    loss = loss.mean()
                    
                elif args.prediction_type == "sample":
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * \
                        F.mse_loss(model_output.float(),
                                   clean_images.float(), reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {args.prediction_type}")
                
                try:
                    out = noise_scheduler.step(
                        model_output.to(torch.float32),
                        timesteps[0].item(),
                        noisy_latents.to(torch.float32),
                        return_dict=True,
                    )
                    pred_0 = out.pred_original_sample
                except:
                    print("exception in noise scheduler step")
                    if args.prediction_type == "epsilon":
                        pred_0 = (noisy_latents - torch.sqrt(1 - alpha_t)
                                    * model_output.float()) / torch.sqrt(alpha_t)
                    elif args.prediction_type == "v_prediction":
                        pred_0 = (alpha_t**0.5) * noisy_latents - (1.0 - alpha_t)**0.5 * model_output.float()
                    else:
                        raise ValueError(
                            f"Unsupported prediction type: {args.prediction_type}")
                    
                with torch.no_grad():
                    img_gt  = vae.decode(clean_latents.float() / vae.config.scaling_factor).sample  # [-1,1]    
                img_rec = vae.decode(pred_0 / vae.config.scaling_factor).sample       # [-1,1]
                
                _lpips_loss   = lpips_loss(img_rec, img_gt).mean()
                _hf_loss      = fft_band_l1_img(img_rec, img_gt, lo=0.15, hi=0.7)
                _tv_loss      = tv_loss(img_rec)
                _checker_loss = checker_loss(img_rec)
                # _lpips_loss ~0.7, hf ~0.08, tv ~5e-6, checker ~6e-8
                # print(_lpips_loss.item(), _tv_loss.item(), _checker_loss.item(), _hf_loss.item())
                w_lpips = 0.05 + 0.15 * min((global_step / 2000), 1)  # 0.05→0.1
                w_hf = 0.05 * min((global_step / 2000), 1)
                loss += _lpips_loss * w_lpips + _hf_loss * w_hf + _tv_loss + _checker_loss * 0.1

                if not all(map(finite_or_none, [img_rec, img_gt])) or not loss_is_finite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.print("⚠️ Non-finite detected, skipping step.")
                    continue
                
                try:
                    accelerator.backward(loss)
                except Exception as e:
                    print(f"exception in backward: {e}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                global_step += 1
                if accelerator.sync_gradients:
                    # Checks if the accelerator has performed an optimization step behind the scene:
                    modules, preclip_norm = topk_grad_modules(model, k=5)
                    agc_(model.time_embedding.parameters(), clip=0.02)
                    agc_(model.conv_out.parameters(), clip=0.02)
                    agc_(model.up_blocks[-1].resnets.parameters(), clip=0.05)
                    clip_by_value_modules(model, modules=["time_emb", "time_embedding", "conv_out", "up_blocks.2.resnets"], clip_val=1.0)
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if args.use_ema:
                        ema_model.step(model.parameters())
                    progress_bar.update(1)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            for name in ["ema", "lpips"]:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith(f"checkpoint-{name}")]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[-1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(
                                        checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                        # save final ema
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-ema-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        # lpips should technically be calculated val set
                        if _best_lpips_loss > _lpips_loss:
                            _best_lpips_loss = _lpips_loss
                            save_path = os.path.join(
                                args.output_dir, f"checkpoint-lpips-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(
                                f"New best lpips {_best_lpips_loss:.4f}, saved to {save_path}")
                        
                    if global_step % args.image_logging_steps == 0:
                            unet = accelerator.unwrap_model(model)

                            if args.use_ema:
                                ema_model.store(unet.parameters())
                                ema_model.copy_to(unet.parameters())

                            generator = torch.Generator(
                                device=pipeline.device).manual_seed(0)
                            # run pipeline in inference (sample random noise and denoise)
                            output = pipeline(
                                # generator=generator,
                                batch_size=1,
                                num_inference_steps=args.ddpm_num_inference_steps,
                                output_type="numpy",
                                batch=test_batch,
                                noise_timesteps=args.ddpm_noise_timesteps,
                                return_dict=True,
                            )
                            
                            if args.use_ema:
                                ema_model.restore(unet.parameters())

                            # denormalize the images and save to tensorboard
                            images = output['images'].permute(0, 2, 3, 1)
                            # denormalize the images and save to tensorboard
                            images = (images.to(torch.float32) / 2 +
                                    0.5).clamp(0, 1).cpu().numpy()
                            images_processed = (images * 255).round().astype("uint8")
                            clean_images = output['clean_images'].permute(0, 2, 3, 1)
                            clean_images = (clean_images.to(torch.float32) / 2 +
                                            0.5).clamp(0, 1).cpu().numpy()
                            clean_images = (clean_images * 255).round().astype("uint8")

                            if args.logger == "tensorboard":
                                if is_accelerate_version(">=", "0.17.0.dev0"):
                                    tracker = accelerator.get_tracker(
                                        "tensorboard", unwrap=True)
                                else:
                                    tracker = accelerator.get_tracker("tensorboard")
                                tracker.add_images(
                                    "test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                            elif args.logger == "wandb":
                                # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                                accelerator.get_tracker("wandb").log(
                                    {"test_samples": [wandb.Image(
                                        img) for img in images_processed], 
                                    "clean_samples": [wandb.Image(
                                        img) for img in clean_images]},
                                    step=global_step,
                                )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], 
                    "step": global_step, "epoch": epoch,
                    "grad_norm": grad_norm.item(),
                    "preclip_norm": preclip_norm,
                    "checker_loss": _checker_loss.item(),
                    "img_loss": _lpips_loss.item(),
                    "tv_loss": _tv_loss.item(),
                    "hf_loss": _hf_loss.item(),
                    "x0_mean": pred_0.mean().item(), # 0
                    "x0_std": pred_0.std().item(),   # stable, no drift
                    **{f"top_grad_{n}": g for n,g in modules}
                    }
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        # batch = {'input': torch.stack([dataset[i]['input'] for i in range(
        #     min(5, len(dataset)))]).to(weight_dtype).to(accelerator.device)}
        if accelerator.is_main_process:
            """
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                generator = torch.Generator(
                    device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                output = pipeline(
                    # generator=generator,
                    batch_size=1,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                    batch=batch,
                    noise_timesteps=args.ddpm_noise_timesteps,
                    return_dict=True,
                )
                
                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images = output['images'].permute(0, 2, 3, 1)
                # denormalize the images and save to tensorboard
                images = (images.to(torch.float32) / 2 +
                          0.5).clamp(0, 1).cpu().numpy()
                images_processed = (images * 255).round().astype("uint8")
                clean_images = output['clean_images'].permute(0, 2, 3, 1)
                clean_images = (clean_images.to(torch.float32) / 2 +
                                0.5).clamp(0, 1).cpu().numpy()
                clean_images = (clean_images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker(
                            "tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images(
                        "test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(
                            img) for img in images_processed], 
                         "clean_samples": [wandb.Image(
                             img) for img in clean_images], "epoch": epoch},
                        step=global_step,
                    )
            """
            
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = UncondLatentDiffusionPipeline(
                    vae=vae,
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
        

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
