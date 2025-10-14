from src.pipeline import UncondLatentDiffusionPipeline
import torch
import os
from src.data import PathologyTrain

device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "checkpoints/ddpm-model-cosine"
data = "checkpoints/ddpm-model-256/image_paths.txt"

num_sampling_steps = 50
noise_timesteps = 500
with open(data, "r") as f:
    paths = f.read().splitlines()
labels = [0]*len(paths)
dataset = PathologyTrain(size=256, paths=paths, labels=labels, debug=False)

# load pipeline
pipeline = UncondLatentDiffusionPipeline.from_pretrained(model_id).to(device)

batch = {'input': torch.stack([dataset[i]['input']
                              for i in range(len(dataset))])}
images = pipeline(num_inference_steps=num_sampling_steps,
                  batch=batch, noise_timesteps=noise_timesteps).images

os.makedirs(model_id + '/images/', exist_ok=True)
for idx, image in enumerate(images):
    image.save(
        f"{model_id}/images/generated_image_e30_{idx}.png")


# save image
# for idx in range(len(dataset)):
#     batch = dataset[idx]
#     batch = {'input': batch['input'].unsqueeze(0).to(device)}
#     image = pipeline(num_inference_steps=num_sampling_steps, batch=batch, noise_timesteps=noise_timesteps).images[0]
#     image.save(f"ddpm-model-256/images/generated_image_{idx}.png")
