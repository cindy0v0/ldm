import os
import gc
import argparse
import numpy as np
from collections import defaultdict
from typing_extensions import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
from PIL import Image

from src.pipeline import UncondLatentDiffusionPipeline
from src.data import PathologyValidation, PathologyTest, PathologyLabels

parser = argparse.ArgumentParser()
parser.add_argument("--lpips_net", type=str,
                    default="vgg", choices=["alex", "vgg"])
parser.add_argument("--model_id", type=str, default="checkpoints/ddpm-model-cosine")
# todo: external set currently only for ocean
parser.add_argument("--external_set", type=str, default=None,
                    choices=["external_set_1", "external_set_2"], help="external sets of ocean")
parser.add_argument("--n_images", type=int, default=8,)
parser.add_argument("--debug", action="store_true",
                    help="debug mode with small dataset")
parser.add_argument("--num_inference_steps", type=int, default=50)
parser.add_argument("--noise_timesteps", type=int, default=350)
parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--device", type=str, default="cpu")

args = parser.parse_args()

torch.cuda.empty_cache()
wdir="./"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = os.path.join(wdir, args.model_id)
args.dtype = torch.float16 if device == "cuda" else torch.float32 


def create_dataset(args):
    dataset = PathologyValidation(size=256, debug=args.debug, num_patches=args.n_images)
    if args.external_set:
        dataset = PathologyTest(size=256, external_set=args.external_set)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return dataset, dataloader



class LDMTest:
    def __init__(self, args, pipeline):
        self.device = args.device
        self.pipe = pipeline.to(self.device)
        self.scheduler = pipeline.scheduler
        self.vae = pipeline.vae
        self.unet = pipeline.unet
        self.scheduler.set_timesteps(args.num_inference_steps)


    @torch.no_grad()    
    def test_schedule(self, args, dataloader):
        batch = next(iter(dataloader))
        noisy_latents, clean_latents = self.pipe.prepare_latents(
            batch, args.noise_timesteps)
        
        decoded_imgs = []
        latents = noisy_latents
        save_freq = args.num_inference_steps // 10
        for t in self.scheduler.timesteps:
            latents = self.scheduler.scale_model_input(latents, t)

            noise_pred = self.unet(
                latents,
                t,
            ).sample
            alpha_t = self.scheduler.alphas_cumprod[t]
            x0_pred = (latents - torch.sqrt(1 - alpha_t)
                                   * noise_pred) / torch.sqrt(alpha_t)
            # log samples
            if t % save_freq == 0 or t == self.scheduler.timesteps[-1]:
                print(f"timestep {t}:")
                decoded_img = self.pipe.decode_latents(x0_pred)
                decoded_imgs.append(decoded_img)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents
            ).prev_sample
            
            del noise_pred, x0_pred
            torch.cuda.empty_cache()
        
        # visualize one long trajectory w decreasing t 
        to_pil = transforms.ToPILImage()
        decoded_imgs = torch.cat(decoded_imgs, dim=0).permute(0, 3, 1, 2)
        images = to_pil(torchvision.utils.make_grid(
            decoded_imgs, nrow=10, normalize=True, value_range=(-1, 1)).cpu())
        
        os.makedirs("test_schedule", exist_ok=True)
        images.save(f"test_schedule/img_{args.num_inference_steps}_{args.noise_timesteps}.png")
        original = batch['input'].squeeze()
        original = to_pil(original).save("test_schedule/original.png")
    
        
if __name__ == "__main__":
    dataset, dataloader = create_dataset(args)
    print("Dataset length: ", len(dataset))
    pipe = UncondLatentDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=args.dtype).to(device)

    tester = LDMTest(args, pipe)
    tester.test_schedule(args, dataloader)
    
    gc.collect()