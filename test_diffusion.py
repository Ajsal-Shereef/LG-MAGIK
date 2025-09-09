import os
import json
import torch
import random

from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

# Paths
checkpoint_dir = "model_weights/instruct-pix2pix-model/checkpoint-20000"  # Your checkpoint directory
val_dir = "data/MiniGrid/validation"  # Validation directory
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load the base Stable Diffusion v1.5 pipeline
base_model = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=False,
)

# Disable the safety checker to avoid NSFW false positives
pipe.safety_checker = None

# Load the fine-tuned U-Net from the checkpoint
unet = UNet2DConditionModel.from_pretrained(
    os.path.join(checkpoint_dir, "unet"),
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
pipe.unet = unet  # Replace the pipeline's U-Net with the fine-tuned one

# Move pipeline to device
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    pipe.enable_xformers_memory_efficient_attention()  # Match training flag if used

# Read metadata
metadata_path = os.path.join(val_dir, "metadata.jsonl")
metadata = []
with open(metadata_path, "r") as f:
    for line in f:
        metadata.append(json.loads(line.strip()))

# Sample 10 random entries
random.seed(42)  # For reproducibility
samples = random.sample(metadata, min(10, len(metadata)))

# Initialize output image dimensions (10 samples, 3 rows: source, target, generated)
img_width, img_height = 256, 256  # Match training resolution
output_image = Image.new("RGB", (img_width * len(samples), img_height * 3))

# Process each sample
for idx, sample in enumerate(samples):
    # Load source and target images
    source_path = os.path.join(val_dir, sample["input_image_file_name"])
    target_path = os.path.join(val_dir, sample["edited_image_file_name"])
    edit_prompt = sample["edit_prompt"]

    source_img = Image.open(source_path).convert("RGB").resize((img_width, img_height), Image.Resampling.LANCZOS)
    target_img = Image.open(target_path).convert("RGB").resize((img_width, img_height), Image.Resampling.LANCZOS)

    # Generate edited image
    generated_img = pipe(
        prompt=edit_prompt,
        image=source_img,
        num_inference_steps=50,  # Balance speed and quality
        image_guidance_scale=1.5,  # Preserve input structure
        guidance_scale=7.5,  # Follow prompt closely
    ).images[0]

    # Place images in the output grid
    output_image.paste(source_img, (idx * img_width, 0))
    output_image.paste(target_img, (idx * img_width, img_height))
    output_image.paste(generated_img, (idx * img_width, 2 * img_height))

# Save the output image
output_path = os.path.join(output_dir, "test_grid.png")
output_image.save(output_path)
print(f"Saved test grid to {output_path}")