import pdb
import lpips
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, roc_curve
import numpy as np
from torchvision import transforms
from collections import defaultdict
from typing_extensions import Dict
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import pytorch_msssim
from torch.utils.data import DataLoader
import torch
from src.pipeline import UncondLatentDiffusionPipeline
import os
import sys
wdir = "/projects/ovcare/users/cindy_shi/ldm/uncond-image-generation-ldm"
sys.path.append(wdir)
sys.path.append(os.path.dirname(wdir))
print(sys.path)


parser = argparse.ArgumentParser()
parser.add_argument("--lpips_net", type=str,
                    default="vgg", choices=["alex", "vgg"])
parser.add_argument("--model_id", type=str, default="checkpoints/ddpm-model")
# todo: external set currently only for ocean
parser.add_argument("--external_set", type=str, default=None,
                    choices=["external_set_1", "external_set_2"], help="external sets of ocean")
# todo: replace with a txt file with test paths
parser.add_argument("--debug", action="store_true",
                    help="debug mode with small dataset")
parser.add_argument("--num_sampling_steps", type=str, default=50)
parser.add_argument("--noise_timesteps", type=str, default=350)
parser.add_argument("--eval_batch_size", type=str, default=16)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = os.path.join(wdir, args.model_id)


def create_dataset():
    from data.data import PathologyValidation, PathologyTest, PathologyLabels
    dataset = PathologyValidation(size=256, debug=args.debug)
    if args.external_set:
        dataset = PathologyTest(size=256, external_set=args.external_set)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return dataset, dataloader


dataset, dataloader = create_dataset()
print("Dataset length: ", len(dataset))
pipe = UncondLatentDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16).to(device)


def compute_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**3
    return size_all_mb


def z_score(scores: torch.Tensor) -> torch.Tensor:
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    mean = scores.mean()
    std = scores.std()
    z_scores = (scores - mean) / (std + 1e-8)
    return z_scores


def slide_z_scores(scores, paths, method="max", labels=None) -> Dict[str, float]:
    # patch to slides
    # average Z-score of all values exceeding the 99’th percentile of the anomaly heatmap
    # or max
    slide_dict = defaultdict(list)
    slide_labels = {}
    for score, path, label in zip(scores, paths, labels):
        slide_id = path.split("/")[int(PathologyLabels.SID_IDX.value)]
        slide_dict[slide_id].append(score.item())
        slide_labels[slide_id] = label

    if method == "max":
        slide_dict = {k: max(v) for k, v in slide_dict.items()}
    elif method == "avg99":
        slide_threshold = {k: torch.tensor(v).kthvalue(
            int(0.99 * len(v))).values.item() for k, v in slide_dict.items()}
        slide_dict = {k: torch.tensor(v)[torch.tensor(
            v) >= slide_threshold[k]].mean().item() for k, v in slide_dict.items()}

    return slide_dict, slide_labels


def dict_to_device(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
ssim_model = pytorch_msssim.MS_SSIM(
    data_range=1.0, size_average=True, channel=3).to(device)
ssim_module = StructuralSimilarityIndexMeasure().to(device)
to_tensor = transforms.ToTensor()
# store z-scores for all test images
#
om_ssims = []
lpips = []
mses = []
paths = []
gts = []

with torch.no_grad():
    for batch in dataloader:
        batch = dict_to_device(batch)
        output = pipe(num_inference_steps=args.num_sampling_steps,
                      noise_timesteps=args.noise_timesteps, batch=batch, return_dict=True)
        images = torch.stack([to_tensor(img)
                             for img in output["images"]]).to(device)
        latents = output["latents"]
        clean_latents = output["clean_latents"]

        ssim = torch.stack([ssim_module(latent.unsqueeze(0), clean_latent.unsqueeze(
            0)) for latent, clean_latent in zip(latents, clean_latents)])
        lpip = lpips_model(images, batch["input"]).mean(dim=[1, 2, 3])
        mse = F.mse_loss(latents, clean_latents,
                         reduction='none').mean(dim=[1, 2, 3])
        if args.debug:
            print((1. - ssim).mean().item(),
                  lpip.mean().item(), mse.mean().item())

        om_ssims.append(torch.tensor([1.]) - ssim.cpu())
        lpips.append(lpip.cpu())
        mses.append(mse.cpu())
        paths.extend(batch["path"])
        gts.append(batch["label"].cpu())

om_ssims = torch.cat(om_ssims)
lpips = torch.cat(lpips)
mses = torch.cat(mses)
gts = torch.cat(gts)

# z-score normalization
om_ssims = z_score(om_ssims)
lpips = z_score(lpips)
mses = z_score(mses)
# combined score
combined = (om_ssims + lpips + mses) / 3.0

# dicts of sid: z-score and sid: label
slide_combined, labels = slide_z_scores(
    combined, paths, method="max", labels=gts)

gts = gts.numpy()
combined = combined.numpy()
slide_combined_scores = list(slide_combined.values())
slide_gt = list(labels.values())


def evaluate(gt, scores):
    roc_auc = roc_auc_score(gt, scores)
    pr_auc = average_precision_score(gt, scores)
    thresh = torch.tensor(scores).kthvalue(
        int(0.7 * len(scores))).values.item()
    bacc = balanced_accuracy_score(
        gt, (np.array(scores) >= thresh).astype(int))
    fpr, tpr, _ = roc_curve(gt, scores)
    fpr95 = fpr[tpr >= 0.95][0]
    print(
        f"Image-level - ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, FPR95: {fpr95:.4f}, BACC (70th percentile): {bacc:.4f}")
    return roc_auc, pr_auc, fpr95, bacc


evaluate(gts, combined)
evaluate(slide_gt, slide_combined_scores)
