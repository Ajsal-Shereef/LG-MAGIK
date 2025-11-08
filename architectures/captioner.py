import os
import random
import shutil
import base64
import httpx
import json
import logging
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Suppress noisy httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

class ImageCaptioner:
    """
    A class to sample images, get captions from an LLM via OpenRouter,
    and save them in a VAE-ready dataset format.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the ImageCaptioner.
        """
        self.cfg = cfg
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it before running.")

        # --- Setup Paths ---
        self.source_path = Path(self.cfg.paths.source_dir)
        self.output_path = Path(self.cfg.paths.output_dir)
        self.output_images_path = self.output_path / "images"
        self.metadata_file_path = self.output_path / "metadata.jsonl"
        self.image_extensions = tuple(self.cfg.data.image_extensions)
        
        # --- Setup API Headers ---
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.cfg.api.referer,
            "X-Title": self.cfg.api.title,
        }

    def _encode_image_to_base64(self, image_path: Path) -> str | None:
        """Encodes an image file to a base64 string."""
        try:
            with Image.open(image_path) as img:
                format = img.format or image_path.suffix.lstrip('.').upper()
                if format == 'JPG':
                    format = 'JPEG'
                
                buffered = BytesIO()
                img.save(buffered, format=format)
                img_bytes = buffered.getvalue()
                
                return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            tqdm.write(f"Error encoding image {image_path.name}: {e}")
            return None

    def _get_llm_caption(self, client: httpx.Client, image_path: Path) -> str | None:
        """Sends the image and prompt to OpenRouter and gets a caption."""
        
        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            return None

        image_format = image_path.suffix.lstrip('.').lower()
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        data_uri = f"data:image/{image_format};base64,{base64_image}"

        payload = {
            "model": self.cfg.model.name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        },
                        {
                            "type": "text",
                            "text": self.cfg.model.prompt
                        }
                    ]
                }
            ],
            "max_tokens": self.cfg.model.max_tokens,
        }

        try:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload
                # Headers and timeout are set on the client itself
            )
            response.raise_for_status()
            data = response.json()
            caption = data['choices'][0]['message']['content']
            return caption.strip()
            
        except httpx.HTTPStatusError as e:
            tqdm.write(f"HTTP error for {image_path.name}: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            tqdm.write(f"Request error for {image_path.name}: {e}")
        except Exception as e:
            tqdm.write(f"Error getting caption for {image_path.name}: {e}")
            
        return None

    def run(self):
        """
        Executes the full captioning and dataset creation process.
        """
        
        # --- 1. Setup Output Directories ---
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_images_path.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {self.output_path.resolve()}")

        # --- 2. Find and Sample Images ---
        print(f"Scanning for images in: {self.source_path.resolve()}")
        all_images = [
            p for p in self.source_path.glob("**/*") 
            if p.is_file() and p.suffix.lower() in self.image_extensions
        ]
        
        if not all_images:
            print("Error: No images found in the source directory.")
            return

        print(f"Found {len(all_images)} total images.")
        
        sample_n = self.cfg.data.sample_n
        if sample_n > len(all_images):
            print(f"Warning: Requested {sample_n} samples, but only {len(all_images)} available. Using all.")
            sample_n = len(all_images)
            
        sampled_images = random.sample(all_images, sample_n)
        print(f"Randomly sampling {len(sampled_images)} images...")

        # --- 3. Process and Save ---
        with httpx.Client(headers=self.headers, timeout=self.cfg.api.timeout) as client:
            with open(self.metadata_file_path, 'a', encoding='utf-8') as f_meta:
                
                for image_path in tqdm(sampled_images, desc="Captioning Images"):
                    
                    caption = self._get_llm_caption(client, image_path)
                    
                    if not caption:
                        tqdm.write(f"Skipping {image_path.name} due to captioning error.")
                        continue
                    
                    # Copy image and handle potential name collisions
                    dest_image_name = image_path.name
                    dest_image_path = self.output_images_path / dest_image_name
                    i = 1
                    while dest_image_path.exists():
                        dest_image_name = f"{image_path.stem}_{i}{image_path.suffix}"
                        dest_image_path = self.output_images_path / dest_image_name
                        i += 1
                        
                    shutil.copy(image_path, dest_image_path)
                    
                    # Write metadata
                    relative_file_name = f"images/{dest_image_name}"
                    metadata_entry = {
                        "file_name": relative_file_name,
                        "text": caption
                    }
                    f_meta.write(json.dumps(metadata_entry) + "\n")

        print("\n--- Processing Complete ---")
        print(f"✅ Sampled images copied to: {self.output_images_path}")
        print(f"✅ Captions saved to: {self.metadata_file_path}")

@hydra.main(version_base=None, config_path=".", config_name="config")
def run_captioning(cfg: DictConfig):
    """
    Hydra entry point for the captioning application.
    """
    print("--- VAE Captioning App ---")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        captioner = ImageCaptioner(cfg)
        captioner.run()
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    run_captioning()