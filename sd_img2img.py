from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

image = Image.open("1.jpg")
prompt = "A blue sleeping bag with a white logo on the front is laid out on a white surface. The sleeping bag is zipped"

out = pipe(prompt=prompt, image=image, strength=0.7, guidance_scale=7.5)
out.images[0].save("output.png")
