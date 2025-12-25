import os
import json
import base64
import shutil
import requests
import hydra
import concurrent.futures
import threading
from pathlib import Path
from dotenv import load_dotenv
from typing import Union, List
from omegaconf import DictConfig

# --- 1. Helper: Base64 Encoder ---
# --- 1. Helper: Base64 Encoder ---
def encode_image(image_input):
    """
    Encodes an image to base64.
    Args:
        image_input (str, Path, PIL.Image.Image, or np.ndarray): The image to encode.
    Returns:
        str: Base64 encoded string.
    """
    from PIL import Image
    import io
    import numpy as np

    if isinstance(image_input, (str, Path)):
        with open(image_input, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    elif isinstance(image_input, np.ndarray):
        # Convert numpy array to PIL Image
        # Assuming image is in [H, W, C] or similar format suitable for PIL
        if image_input.dtype != np.uint8:
             image_input = (image_input).astype(np.uint8)
        image = Image.fromarray(image_input)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    elif isinstance(image_input, Image.Image):
        buffered = io.BytesIO()
        image_input.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

# --- 2. Helper: Text Parser ---
def split_gptoss_analysis_final(content: str):
    import re
    final_match = re.search(r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>", content, re.DOTALL)
    analysis_match = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", content, re.DOTALL)
    
    final_text = final_match.group(1).strip() if final_match else content
    analysis_text = analysis_match.group(1).strip() if analysis_match else None
    
    return analysis_text, final_text

# --- 3. Query Function ---
def query_llm(system: str, prompt: Union[str, List[dict]], api_key: str, cfg: DictConfig = None, mode: str = None, pipeline: str = None, temperature: float = None) -> tuple[str, dict]:
    """
    Queries the LLM. 
    Accepts specific args (mode, pipeline, temperature) OR a Hydra cfg object.
    """
    # Fallback to cfg if explicit args are not provided
    if mode is None and cfg is not None:
        mode = cfg.model.mode
    if pipeline is None and cfg is not None:
        pipeline = cfg.model.name
    if temperature is None and cfg is not None:
        temperature = cfg.model.temperature
        
    if mode is None:
        raise ValueError("Mode must be provided either explicitly or via cfg.")

    if mode == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:3000", 
        }
        
        structured_system_prompt = f"""{system}
        """
        
        payload = {
            "model": pipeline,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": structured_system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Handle potential empty choices from API errors
            if not result.get("choices"):
                raise RuntimeError(f"OpenRouter returned no choices: {result}")
                
            final = result["choices"][0]["message"]["content"]
            return final
        else:
            raise RuntimeError(f"OpenRouter failed: {response.status_code}: {response.text}")
    
    elif mode == "huggingface":
        return "HF Mode not implemented for Vision in this snippet", {}

# --- 4. Main Processing Logic ---
def process_dataset(cfg: DictConfig, api_key: str):
    # Extract paths from Hydra config
    source_path = Path(cfg.paths.source_dir)
    output_path = Path(cfg.paths.output_dir)
    images_output_path = output_path / "images"
    metadata_path = output_path / "metadata.jsonl"

    # Create directories
    images_output_path.mkdir(parents=True, exist_ok=True)

    # --- RESUME LOGIC ---
    processed_files = set()
    if metadata_path.exists():
        print(f"Metadata file found. Checking for existing entries...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "file_name" in entry:
                        processed_files.add(entry["file_name"])
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(processed_files)} already processed images.")

    # Collect source files
    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = [f for f in source_path.iterdir() if f.suffix.lower() in valid_extensions]

    # --- SORTING LOGIC ---
    def sort_key(f):
        try:
            return int(f.stem)
        except ValueError:
            return float('inf') 
            
    all_files.sort(key=sort_key)

    print(f"Found {len(all_files)} total images in source.")

    # Lock for writing to the metadata file
    write_lock = threading.Lock()

    def process_single_image(file_path):
        filename = file_path.name
        
        # Double check inside the thread (though we filter before submitting too)
        if filename in processed_files:
            return

        print(f"Processing: {filename}...")

        try:
            base64_image = encode_image(file_path)

            vision_prompt = [
                {"type": "text", "text": "Describe this image for a text-to-image training dataset."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]

            # Pass cfg to query_llm to access model configs
            caption = query_llm(
                system=cfg.prompt.system,
                prompt=vision_prompt,
                api_key=api_key,
                cfg=cfg, 
            )

            destination_path = images_output_path / filename
            shutil.copy(file_path, destination_path)

            entry = {
                "file_name": filename,
                "text": caption
            }
            
            with write_lock:
                with open(metadata_path, "a", encoding="utf-8") as jsonl_file:
                    jsonl_file.write(json.dumps(entry) + "\n")
                    jsonl_file.flush()
            
            print(f"   -> Completed {filename}")

        except Exception as e:
            print(f"   -> Error processing {filename}: {e}")
            # raise e # Optional: re-raise if you want to stop everything, but usually better to log and continue

    # Use ThreadPoolExecutor for concurrent processing
    batch_size = cfg.model.get("batch_size", 10) # Default to 10 if not set
    print(f"Starting processing with batch size (concurrency): {batch_size}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for file_path in all_files:
            if file_path.name in processed_files:
                continue
            futures.append(executor.submit(process_single_image, file_path))
        
        # Wait for all futures to complete (optional, context manager does this anyway)
        concurrent.futures.wait(futures)

# --- 5. Hydra Entry Point ---
@hydra.main(version_base=None, config_path="config", config_name="captioner")
def main(cfg: DictConfig):
    # Load environment variables
    load_dotenv(dotenv_path="config/.env")
    
    # Get API Key based on the name defined in config
    api_key_name = cfg.api.env_var_name
    api_key = os.getenv('OPEN_ROUTER_API_KEY')
    
    if api_key:
        process_dataset(cfg, api_key)
    else:
        print(f"Error: Environment variable {api_key_name} not found.")

if __name__ == "__main__":
    main()