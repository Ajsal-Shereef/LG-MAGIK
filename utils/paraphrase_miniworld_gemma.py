import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from tqdm import tqdm
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf

# Allow importing project modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

SYSTEM_PROMPT = """You are an expert descriptive annotator for 3D reinforcement learning environments.
Your goal is to generate natural, diverse, human-written qualitative scene captions by synthesizing two provided inputs:

INPUTS PROVIDED TO YOU:
1. Scene Image: The first-person RGB camera observation showing the 3D room, objects, floor, walls, and sky.
2. Template Observation: A programmatic description containing ground-truth object identities, colors, and exact numerical measurements (distance in units, angle in degrees).

HOW TO COMBINE THEM:
- Always cross-verify the template description against the image before outputting the final caption.
- Use the Template Observation as ground truth for what objects actually exist, preventing hallucinations or missed objects.
- Use the Scene Image to verify what is actually visually visible (especially the sky, walls, and relative layout).
- Translate exact numerical measurements into natural, intuitive qualitative spatial descriptions.

CRITICAL RULES:
1. REMOVE ALL NUMERICAL MEASUREMENTS: Strip out all exact numbers, units, and degree measurements (e.g. remove "1.2 units", "24.0 degrees", "4.5 units").
2. CONVERT TO QUALITATIVE SPATIAL DESCRIPTIONS:
   - Distance:
     * < 2.0 units -> "nearby", "in the foreground", "close to the agent"
     * 2.0 - 3.8 units -> "ahead", "at a moderate distance", "midway across the room"
     * > 3.8 units -> "far away", "in the distance", "against the back wall"
   - Direction / Perspective:
     * < 10 degrees -> "straight ahead", "centered", "directly in front"
     * 10 - 45 degrees -> "to the left", "to the right", "forward-left", "forward-right"
     * > 45 degrees -> "far to the left", "far to the right"
3. VISUAL VERIFICATION OF SKY & WALLS (CRITICAL):
   - The template observation often mentions "under a blue sky" by default, even when the sky is NOT visible in the image.
   - When the agent is positioned near a wall or facing a wall up close, the gray wall may fill the upper frame completely, meaning NO sky is visible.
   - Always check the image: ONLY describe the blue sky if it is genuinely visible above the walls. If only the walls and floor are visible, do NOT mention the sky!
4. STRICT OBJECT ACCURACY: Strictly preserve object categories, colors, walls, and flooring as shown in the image and template. Never invent or omit objects.
5. NATURAL DIVERSITY: Vary the sentence structure and vocabulary across samples. Avoid repetitive robotic openings like "The image depicts" or "A [object] is visible at".
6. Make sure your caption doesn't contain any of the following:
   - Single word garbage: "thought", ".thought", ".", "sky.", "light"
   - Code-like garbage and control token fragments with no scene caption (e.g. "\u2558{", "\u2558thought", "\u2558.png")
   - Pure non-English (Chinese / CJK) captions
   - Very short lowercase fragments (<=2 words, no uppercase start)
   - Fragments like "2.0 units" or measurement fragments
   - Unicode-only garbage text
   - Incomplete scene descriptions (missing floor and object words)
   - Empty/whitespace-only text
   - Leading/trailing whitespace and flattens newlines
   - Collapses multiple internal spaces into single spaces
   - Leading control characters (\u2558, \u255b, etc.)
   - Leading unicode garbage characters (cuneiform and high-unicode)
   - Leading dots and ellipsis (". A blue box..." -> "A blue box...")
   - Leading HTML/special token tags like "<unused2682>"
   - leading non-ASCII foreign word fragments (e.g. 'éparsément', Bengali 'রিহাম', 'অচিতের')
   - Gemma hallucinated "Ja" prefixes ("Ja blue box..." -> "A blue box...", "Ja-colored box..." -> "A blue box...", "Ja-colored walls..." -> "Gray walls...")
   - Gemma hallucinated "Jasmin" prefixes ("Jasmin-green..." -> "Green...", "Jasmin-blue box..." -> "A blue box...")
   - Hallucinated proper names ("Johnson's room...", "Johnson-colored walls...", "Williams-style...", "Wilson-style...")
   - "Iseries of..." -> "A series of..." and strips "Iser"
   - Capitalized and lowercase garbage tokens ("RageBet", "Sakammak", "SetPenis", "Dtdsoftware", "Feder...", "Tema", "OpA", "opside", "Tingered", "Tingerly", "Nuss", etc.)
   - Leading numbering like "1.", "7." at start of captions
   - Capitalization typos at start ("VEry" -> "Very", "VAst" -> "Vast", "THe" -> "The", "WIthin" -> "Within", "Openeing" -> "Opening", "Situtated" -> "Situated")
   - Missing space after sentence period (e.g. "grass.A blue box" -> "grass. A blue box")
   - Auto-capitalizes first letter if lowercase
   - Appends missing trailing period if absent
7. OUTPUT FORMAT: Output ONLY the final qualitative caption directly. No quotes, no preamble, and no conversational filler."""

def build_prompt_text(raw_text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Template observation:\n{raw_text}\n\n"
        "Qualitative scene caption:"
    )

def clean_generated_caption(text: str) -> str:
    # Truncate at turn separator or eos/pad if present in decoded text
    for stop_seq in ["<turn|>", "<|turn>", "<eos>", "<pad>", "<|channel|>"]:
        if stop_seq in text:
            text = text.split(stop_seq)[0]
    
    # Remove any internal channel or thought blocks if present
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel\|>analysis.*?<\|end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel\|>final<\|message\|>", "", text)
    text = re.sub(r"<\|return\|>", "", text)
    
    # Strip any special token strings
    for token in ["<eos>", "<turn|>", "<|turn>", "<pad>"]:
        text = text.replace(token, "")
        
    # Remove leading/trailing quotes or conversational prefixes
    text = text.strip().strip('"\'')
    if text.lower().startswith("paraphrased caption:"):
        text = text[len("paraphrased caption:"):].strip()
    if text.lower().startswith("paraphrase:"):
        text = text[len("paraphrase:"):].strip()
    if text.lower().startswith("caption:"):
        text = text[len("caption:"):].strip()
        
    # If multiple lines, take the first non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        text = lines[0]
    return text.strip().strip('"\'')

def setup_image_links(source_images_dir: Path, output_images_dir: Path, link_mode: str = "symlink"):
    if link_mode == "symlink":
        if output_images_dir.is_symlink() or output_images_dir.exists():
            return
        print(f"[INFO] Symlinking directory {source_images_dir} -> {output_images_dir}...")
        output_images_dir.symlink_to(source_images_dir.resolve(), target_is_directory=True)
        print("[INFO] Directory symlink complete.")
    elif link_mode == "copy":
        output_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Copying images from {source_images_dir} to {output_images_dir}...")
        for src_file in source_images_dir.glob("*.png"):
            dst_file = output_images_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
        print("[INFO] Image copying complete.")

def run_paraphrasing(
    source_dir: str,
    output_dir: str,
    model_id: str = "google/gemma-4-12B-it",
    batch_size: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 80,
    limit: Optional[int] = None,
    link_mode: str = "symlink",
    log_interval: int = 100
):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    source_metadata_file = source_path / "metadata.jsonl"
    source_images_dir = source_path / "images"
    output_metadata_file = output_path / "metadata.jsonl"
    output_images_dir = output_path / "images"

    output_path.mkdir(parents=True, exist_ok=True)

    if not source_metadata_file.exists():
        raise FileNotFoundError(f"Source metadata file not found at {source_metadata_file}")

    # 1. Setup images (symlink or copy)
    if source_images_dir.exists():
        setup_image_links(source_images_dir, output_images_dir, link_mode=link_mode)

    # 2. Check existing processed items for resume capability
    processed_files = set()
    if output_metadata_file.exists():
        with open(output_metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "file_name" in entry:
                        processed_files.add(entry["file_name"])
                except json.JSONDecodeError:
                    continue
        print(f"[INFO] Found {len(processed_files)} already processed entries. Resuming from next entry...")

    # 3. Read pending entries
    entries_to_process = []
    with open(source_metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                fname = entry.get("file_name", "")
                if fname not in processed_files:
                    entries_to_process.append(entry)
                    if limit is not None and len(entries_to_process) >= limit:
                        break
            except json.JSONDecodeError:
                continue

    print(f"[INFO] Total entries remaining to process: {len(entries_to_process)}")
    if not entries_to_process:
        print("[INFO] All entries are already processed! Nothing to do.")
        return

    # 4. Load Multimodal Model & Processor
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[INFO] Loading multimodal processor and model ({model_id}) onto GPU...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    print("[INFO] Model loaded successfully. Starting multimodal batch generation...")

    # 5. Process in batches
    num_batches = (len(entries_to_process) + batch_size - 1) // batch_size
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_metadata_file, "a", encoding="utf-8") as out_f:
        for b_idx in tqdm(range(num_batches), desc="Paraphrasing MiniWorld Captions (Multimodal)"):
            batch_entries = entries_to_process[b_idx * batch_size : (b_idx + 1) * batch_size]
            prompts = []
            images = []

            for entry in batch_entries:
                raw_text = entry.get("text", "")
                prompt_text = build_prompt_text(raw_text)

                # Locate image
                img_rel = entry.get("file_name", "")
                img_path = source_path / img_rel
                if not img_path.exists():
                    img_path = source_images_dir / Path(img_rel).name

                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        print(f"[WARN] Failed to open {img_path}: {e}, using blank image.")
                        img = Image.new("RGB", (64, 64), color="black")
                else:
                    img = Image.new("RGB", (64, 64), color="black")

                images.append([img])

                msgs = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                formatted_prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                prompts.append(formatted_prompt)

            # Tokenize & encode image batch
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p
                )

            # Decode each sample in the batch
            for idx, entry in enumerate(batch_entries):
                out_tokens = outputs[idx][inputs["input_ids"].shape[-1] :]
                # skip_special_tokens=True cleanly strips <pad>, <eos>, and other special tokens
                decoded_text = processor.decode(out_tokens, skip_special_tokens=True)
                clean_text = clean_generated_caption(decoded_text)

                out_entry = {
                    "file_name": entry["file_name"],
                    "text": clean_text
                }
                out_f.write(json.dumps(out_entry) + "\n")
                out_f.flush()

    print(f"\n[INFO] Paraphrasing completed successfully! Results saved to {output_metadata_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Paraphrase MiniWorld captions using local Gemma 4 12B (Multimodal)")
    parser.add_argument("--source_dir", type=str, default=None, help="Source data folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Destination folder")
    parser.add_argument("--model_id", type=str, default=None, help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for parallel inference")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens to generate")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples (for testing)")
    parser.add_argument("--link_mode", type=str, default=None, choices=["symlink", "copy"], help="Image linking strategy")
    return parser.parse_known_args()

@hydra.main(version_base=None, config_path="../config", config_name="paraphrase_miniworld")
def hydra_main(cfg: DictConfig):
    # Support overriding via direct hydra or standard config
    source_dir = cfg.paths.get("source_dir", "data/Variable_data_MiniWorld")
    output_dir = cfg.paths.get("output_dir", "data/Qualitative_data_MiniWorld")
    link_mode = cfg.paths.get("link_mode", "symlink")

    model_id = cfg.model.get("model_id", "google/gemma-4-12B-it")
    batch_size = cfg.model.get("batch_size", 16)
    temperature = cfg.model.get("temperature", 0.7)
    top_p = cfg.model.get("top_p", 0.9)
    max_new_tokens = cfg.model.get("max_new_tokens", 80)

    limit = cfg.options.get("limit", None)
    log_interval = cfg.options.get("log_interval", 100)

    run_paraphrasing(
        source_dir=source_dir,
        output_dir=output_dir,
        model_id=model_id,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        limit=limit,
        link_mode=link_mode,
        log_interval=log_interval
    )

if __name__ == "__main__":
    has_argparse_flags = any(arg.startswith("--") and not arg.startswith("--hydra") and not arg.startswith("--cfg") for arg in sys.argv[1:])
    if has_argparse_flags:
        cli_args, unknown = parse_args()
        run_paraphrasing(
            source_dir=cli_args.source_dir or "data/Variable_data_MiniWorld",
            output_dir=cli_args.output_dir or "data/Qualitative_data_MiniWorld",
            model_id=cli_args.model_id or "google/gemma-4-12B-it",
            batch_size=cli_args.batch_size or 16,
            temperature=cli_args.temperature if cli_args.temperature is not None else 0.7,
            top_p=cli_args.top_p if cli_args.top_p is not None else 0.9,
            max_new_tokens=cli_args.max_new_tokens or 80,
            limit=cli_args.limit,
            link_mode=cli_args.link_mode or "symlink"
        )
    else:
        hydra_main()
