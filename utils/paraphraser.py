import os
import json
import shutil
import requests
import hydra
import concurrent.futures
import threading
from pathlib import Path
from dotenv import load_dotenv
from typing import Union, List
from omegaconf import DictConfig

def query_nvidia(system: str, prompt: str, api_key: str, model_name: str, temperature: float, base_url: str = None) -> str:
    if not api_key:
        raise ValueError("Nvidia API key not found")
        
    invoke_url = base_url or "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False
    }
    
    if system:
        payload["messages"].insert(0, {"role": "system", "content": system})

    response = requests.post(invoke_url, headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        if not result.get("choices"):
             raise RuntimeError(f"Nvidia API returned no choices: {result}")
        return result["choices"][0]["message"]["content"]
    else:
         raise RuntimeError(f"Nvidia API failed: {response.status_code}: {response.text}")

def query_anthropic(system: str, prompt: str, api_key: str, model_name: str, temperature: float) -> str:
    if not api_key:
        raise ValueError("Anthropic API key not found")
        
    invoke_url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "max_tokens": 512,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    if system:
        payload["system"] = system

    response = requests.post(invoke_url, headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        if not result.get("content"):
             raise RuntimeError(f"Anthropic API returned no content: {result}")
        return result["content"][0]["text"]
    else:
         raise RuntimeError(f"Anthropic API failed: {response.status_code}: {response.text}")

def query_openrouter(system: str, prompt: str, api_key: str, model_name: str, temperature: float) -> str:
    if not api_key:
        raise ValueError("OpenRouter API key not found")
        
    invoke_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/lg-magik",
        "X-Title": "LG-MAGIK",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False
    }
    
    if system:
        payload["messages"].insert(0, {"role": "system", "content": system})

    response = requests.post(invoke_url, headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        if not result.get("choices"):
             raise RuntimeError(f"OpenRouter API returned no choices: {result}")
        return result["choices"][0]["message"]["content"]
    else:
         raise RuntimeError(f"OpenRouter API failed: {response.status_code}: {response.text}")

def query_llm_with_fallback(system: str, prompt: str, keys: dict, cfg: DictConfig) -> str:
    temperature = cfg.model.get("temperature", 0.7)
    base_url = cfg.model.get("base_url", None)
    
    errors = []

    # 1. Anthropic Main
    try:
        if keys.get("anthropic"):
            return query_anthropic(system, prompt, keys.get("anthropic"), cfg.model.anthropic_main, temperature)
    except Exception as e:
        errors.append(f"Anthropic Main ({cfg.model.anthropic_main}): {e}")
        print(f"      [Fallback] Anthropic Main failed: {e}")

    # 2. Anthropic Alt
    try:
        if keys.get("anthropic"):
            return query_anthropic(system, prompt, keys.get("anthropic"), cfg.model.anthropic_alt, temperature)
    except Exception as e:
        errors.append(f"Anthropic Alt ({cfg.model.anthropic_alt}): {e}")
        print(f"      [Fallback] Anthropic Alt failed: {e}")

    # 3. Nvidia Main
    try:
        return query_nvidia(system, prompt, keys.get("nvidia"), cfg.model.nvidia_main, temperature, base_url)
    except Exception as e:
        errors.append(f"Nvidia Main ({cfg.model.nvidia_main}): {e}")
        print(f"      [Fallback] Nvidia Main failed: {e}")
        
    # 2. Nvidia Alt
    try:
        return query_nvidia(system, prompt, keys.get("nvidia"), cfg.model.nvidia_alt, temperature, base_url)
    except Exception as e:
        errors.append(f"Nvidia Alt ({cfg.model.nvidia_alt}): {e}")
        print(f"      [Fallback] Nvidia Alt failed: {e}")

    # 3. OpenRouter Main
    try:
        return query_openrouter(system, prompt, keys.get("openrouter"), cfg.model.openrouter_main, temperature)
    except Exception as e:
        errors.append(f"OpenRouter Main ({cfg.model.openrouter_main}): {e}")
        print(f"      [Fallback] OpenRouter Main failed: {e}")

    # 4. OpenRouter Alt
    try:
        return query_openrouter(system, prompt, keys.get("openrouter"), cfg.model.openrouter_alt, temperature)
    except Exception as e:
        errors.append(f"OpenRouter Alt ({cfg.model.openrouter_alt}): {e}")
        print(f"      [Fallback] OpenRouter Alt failed: {e}")
        
    raise RuntimeError(f"All LLM API calls failed. Errors:\n" + "\n".join(errors))

def process_dataset(cfg: DictConfig, keys: dict):
    source_path = Path(cfg.paths.source_dir)
    output_path = Path(cfg.paths.output_dir)
    images_output_path = output_path / "images"
    metadata_path = output_path / "metadata.jsonl"

    images_output_path.mkdir(parents=True, exist_ok=True)

    processed_files = set()
    if metadata_path.exists():
        print(f"Metadata file found. Checking for existing entries...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "file_name" in entry:
                        processed_files.add(entry["file_name"].split("/")[-1])
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(processed_files)} already processed entries.")

    source_metadata_path = source_path.parent / "metadata.jsonl"
    if not source_metadata_path.exists():
        print(f"Error: Source metadata not found at {source_metadata_path}")
        return

    entries = []
    print(f"Loading source metadata from {source_metadata_path}...")
    with open(source_metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                fname = Path(entry.get("file_name", "")).name
                if fname:
                    entries.append((fname, entry.get("text", "")))
            except json.JSONDecodeError:
                continue

    write_lock = threading.Lock()

    def process_single_entry(entry_data):
        filename, template_text = entry_data
        
        if filename in processed_files:
            return

        print(f"Processing: {filename}...")

        try:
            if template_text:
                prompt_text = f"Paraphrase this following template description into a natural, diverse human-like caption.\n\nOriginal template caption:\n{template_text}"
            else:
                prompt_text = "No objects are in the scene. Provide a brief empty scene description for a 2D gridworld."

            caption = query_llm_with_fallback(
                system=cfg.prompt.system,
                prompt=prompt_text,
                keys=keys,
                cfg=cfg, 
            )

            # Copy image to keep dataset structure exactly parallel
            source_file = source_path / filename
            if source_file.exists():
                destination_path = images_output_path / filename
                shutil.copy(source_file, destination_path)
            
            entry = {
                "file_name": f"images/{filename}",
                "text": caption.strip(),
            }
            
            with write_lock:
                with open(metadata_path, "a", encoding="utf-8") as jsonl_file:
                    jsonl_file.write(json.dumps(entry) + "\n")
                    jsonl_file.flush()
            
            print(f"   -> Completed {filename}")

        except Exception as e:
            print(f"   -> Error processing {filename}: {e}")

    batch_size = cfg.model.get("batch_size", 20)
    print(f"Starting processing with batch size: {batch_size}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for entry_data in entries:
            if entry_data[0] in processed_files:
                continue
            futures.append(executor.submit(process_single_entry, entry_data))
        
        concurrent.futures.wait(futures)

@hydra.main(version_base=None, config_path="config", config_name="paraphraser")
def main(cfg: DictConfig):
    load_dotenv(dotenv_path="config/.env")
    anthropic_api_key = os.getenv(cfg.api.get("anthropic_env_var", "ANTHROPIC_API_KEY"))
    nvidia_api_key = os.getenv(cfg.api.get("nvidia_env_var", "NVIDIA_API_KEY"))
    openrouter_api_key = os.getenv(cfg.api.get("openrouter_env_var", "OPENROUTER_API_KEY"))
    
    keys = {
        "anthropic": anthropic_api_key,
        "nvidia": nvidia_api_key,
        "openrouter": openrouter_api_key
    }
    
    if not anthropic_api_key and not nvidia_api_key and not openrouter_api_key:
        print("Error: No API keys were found.")
        return
        
    process_dataset(cfg, keys)

if __name__ == "__main__":
    main()
