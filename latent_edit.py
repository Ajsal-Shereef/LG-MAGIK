import pickle
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Initialize the InstructPix2Pix pipeline
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

def edit_minigrid_image(observation, prompt="change the purple box to blue key", orig_obs=None):
    """
    Edit an RGB MiniGrid observation using InstructPix2Pix.
    
    Parameters:
        observation (np.ndarray): RGB image array of shape (height, width, 3)
        prompt (str): Text prompt for editing (default: "make the purple key into red ball")
        orig_obs (np.ndarray, optional): Original MiniGrid observation with (obj_type, color, state)
    
    Returns:
        np.ndarray: Edited RGB image array
        np.ndarray (optional): Updated MiniGrid observation if orig_obs is provided
    """
    
    # Convert to PIL Image
    pil_image = observation
    
    pil_image.save("original_frame.png")
    # Perform editing with InstructPix2Pix
    edited_image = pipe(
        prompt=prompt,
        image=pil_image,
        num_inference_steps=10,
        guidance_scale=7.5,
        image_guidance_scale=1.5,  # Balance between original image and prompt
    ).images[0]
    edited_image.save("edited_frame.png")
    edited_array = np.array(edited_image)
    
    # If orig_obs is provided, update the MiniGrid observation format
    if orig_obs is not None:
        new_observation = np.zeros_like(orig_obs)
        for i in range(orig_obs.shape[0]):
            for j in range(orig_obs.shape[1]):
                if (orig_obs[i, j, 0] == OBJECT_TO_IDX['key'] and 
                    orig_obs[i, j, 1] == COLOR_TO_IDX['purple']):
                    new_observation[i, j] = [OBJECT_TO_IDX['ball'], COLOR_TO_IDX['red'], 0]
                else:
                    new_observation[i, j] = orig_obs[i, j]
        return edited_array, new_observation
    
    return edited_array

# Example usage 
image = Image.open("data/MiniGrid/training/images/edited_000000.png")
edited_rgb = edit_minigrid_image(image)