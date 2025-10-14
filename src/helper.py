import torch
import torch.nn as nn
from standardizer import ChannelStandardize

@torch.no_grad()
def compute_running_channel_stats(vae, dataloader, device, max_batches=1200):
    sf = vae.config.scaling_factor
    sh = getattr(vae.config, "shift_factor", 0.0)
    C = vae.config.latent_channels
    n = torch.zeros(C, device=device)
    mean = torch.zeros(C, device=device)
    M2 = torch.zeros(C, device=device)

    def update(x):  # x: [B,C,H,W]
        B, C, H, W = x.shape
        x = x.permute(1,0,2,3).contiguous().view(C, -1)   # [C, B*H*W]
        cnt = x.size(1)
        n.add_(cnt)
        delta = x.mean(dim=1) - mean
        mean.add_(delta * (cnt / n))
        # per-channel var increment
        x_c = x - mean.unsqueeze(1)
        M2.add_((x_c**2).sum(dim=1))

    for i, batch in enumerate(dataloader):
        if i >= max_batches: break
        x = batch["input"].to(device)
        posterior = vae.encode(x)
        if hasattr(posterior, "latent_dist"):
            z_raw = posterior.latent_dist.sample()
        else:
            z_raw = posterior.latents
        z = (z_raw - sh) * sf
        update(z)

    var = M2 / (n.clamp_min(1) - 1)
    std = var.clamp_min(1e-6).sqrt()
    return mean.detach().cpu(), std.detach().cpu()


if __name__ == "__main__":
    from diffusers import AutoencoderKL, VQModel
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from data import PathologyTrain

    vqvae = VQModel.from_pretrained(
        "CompVis/ldm-celebahq-256", subfolder="vqvae").to("cuda")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", subfolder="vae").to("cuda")
    
    dataset = PathologyTrain(size=256)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=4)

    mean, std = compute_running_channel_stats(vae, dataloader, device="cuda")
    print("Channel mean:", mean)
    print("Channel std: ", std)

    chan_std = ChannelStandardize(mean, std)
    torch.save(chan_std.state_dict(), "chan_std.pth")
    
    mean, std = compute_running_channel_stats(vqvae, dataloader, device="cuda")
    print("Channel mean:", mean)
    print("Channel std: ", std)