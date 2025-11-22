import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid
from diffusers import DDPMScheduler

from utils import parse_int_list
from pipeline import UncondLatentDiffusionPipeline
from data import PathologyBase

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize reconstruction process of latent diffusion model")
    parser.add_argument(
        "--model_id", type=str, default="checkpoints/ddpm-model-cnorm-snr", help="Pretrained model ID or path.")
    parser.add_argument(
        "--noise_timesteps", type=parse_int_list, default="50,100,200", help="Comma-separated list of noise timesteps to visualize.")
    parser.add_argument(
        "--results_dir", type=str, default="results")
    parser.add_argument(
        "--train", action='store_true', help='Whether to use training data for visualization.')
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for dataloader.")
    parser.add_argument(
        "--beta_end", type=float, default=0.01, help="Maximum beta value for the noise schedule.")
    return parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()
noise_timesteps = args.noise_timesteps
model_id = args.model_id
num_inference_steps = 50

train_dataset = PathologyBase('train', size=256, color_norm=True, n_samples=6, random_subsample=False)
val_dataset = PathologyBase('validation', size=256, color_norm=True, n_samples=6)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

pipe = UncondLatentDiffusionPipeline.from_pretrained(model_id).to(device)
config = pipe.scheduler.config
config["beta_end"] = args.beta_end
config["beta_schedule"] = "scaled_linear"
pipe.scheduler = DDPMScheduler.from_config(config)
pipe.scheduler.set_timesteps(num_inference_steps)

def save_grid(grid: torch.Tensor, save_pth: str):
    np_grid = ((grid.permute(1, 2, 0).numpy() + 1.0) / 2 * 255).clip(0, 255).astype(np.uint8)
    pil_grid = Image.fromarray(np_grid)
    pil_grid.save(save_pth)

@torch.no_grad()
def reconstruct_images(batch, noise_timestep) -> tuple[list[torch.Tensor], torch.Tensor]:
    images = []
    
    latents, clean_latents = pipe.prepare_latents(
        batch, noise_timesteps=noise_timestep)
    clean_image = pipe.decode_latents(clean_latents)
    noised_image = pipe.decode_latents(latents)

    for t in pipe.scheduler.timesteps:
        latents = pipe.scheduler.scale_model_input(latents, t)

        noise_pred = pipe.unet(
            latents,
            t,
        ).sample

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(
            noise_pred, t, latents,
        ).prev_sample

        image = pipe.decode_latents(latents)
        # clean_image = pipe.decode_latents(clean_latents)
        images.append(image.cpu())
        del image, noise_pred
    
    torch.cuda.empty_cache()
    # images = torch.cat(images, dim=0)
    return images, clean_image.cpu(), noised_image.cpu()

def visualize_noise_latents_recon():
    all_images, all_clean_images, all_noised_images = [], [], []
    recon_pth = []
    for noise_timestep in noise_timesteps:
        images, clean_images, noised_images = reconstruct_images(batch, noise_timestep=noise_timestep)
        all_images.append(images[-1]); all_clean_images.append(clean_images); all_noised_images.append(noised_images)
        recon_pth.append(images)
    
    all_images = torch.cat(all_images, dim=0); all_clean_images = torch.cat(all_clean_images, dim=0); all_noised_images = torch.cat(all_noised_images, dim=0)
    grid = torch.stack([all_clean_images, all_noised_images, all_images], dim=1).reshape(-1, 3, 256, 256)
    grid = make_grid(grid, nrow=3)
    
    save_pth=f"{args.results_dir}/{model_id.split('/')[-1]}_{data}_noised_reconstruction_{'_'.join(map(str, noise_timesteps))}.png"
    save_grid(grid, save_pth=save_pth)
    
    recon_pth = torch.cat(recon_pth, dim=0)
    grid = make_grid(recon_pth, nrow=num_inference_steps)
    save_pth=f"{args.results_dir}/{model_id.split('/')[-1]}_{data}_full_reconstruction_trajectory.png"
    save_grid(grid, save_pth=save_pth)

def visualize_recon_timesteps(batch, noise_timesteps=[100, 200, 350, 500, 750, 1000], save_pth="results/reconstruction_timesteps.png"):
    recon_images = []
    for noise_timestep in noise_timesteps:
        images, clean_recon_images, _ = reconstruct_images(batch, noise_timestep)
        recon_images.append(images[-1])
    recon_images = torch.stack([clean_recon_images] + recon_images, dim=1).reshape(-1, 3, 256, 256)
    grid = make_grid(recon_images, nrow=len(noise_timesteps) + 1)
    
    save_grid(grid, save_pth=save_pth)    
    
    return grid

def visualize_vae_recon(batch, save_pth="results/vae_reconstruction.png"):
    clean_images = batch['input']
    posterior = pipe.vae.encode(clean_images.to(device))
    clean_latent = posterior.latent_dist.sample() if hasattr(posterior, 'latent_dist') else posterior.latents
    clean_recon = pipe.vae.decode(clean_latent, return_dict=False)[0].cpu()
    grid = torch.stack([clean_images, clean_recon], dim=1).reshape(-1, 3, 256, 256)
    grid = make_grid(grid, nrow=2)
    
    save_grid(grid, save_pth=save_pth)

def plot_alphas_cumprod():
    alphas_cumprod = pipe.scheduler.alphas_cumprod.cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.plot(alphas_cumprod, label='alphas_cumprod')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.title('Alphas Cumprod over Timesteps')
    plt.legend()
    plt.grid()
    plt.savefig(f"{args.results_dir}/alphas_cumprod_scaled_linear.png")    
    

    
if __name__ == "__main__":
    if args.train:
        batch = next(iter(train_dataloader))
        data = "train"
    else:
        batch = next(iter(val_dataloader))
        data = "validation"

    save_pth=f"{args.results_dir}/{os.path.basename(model_id)}_{data}_reconstruction_steps_{'_'.join(map(str, noise_timesteps))}.png"
    visualize_recon_timesteps(batch, 
                              noise_timesteps=noise_timesteps,
                              save_pth=save_pth)
    # dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=6, shuffle=False)
    # batch = next(iter(dataloader))