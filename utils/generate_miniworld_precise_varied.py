#!/usr/bin/env python3
"""
Generate a new MiniWorld dataset combining the linguistic variability
from data/MiniWorldNoisy and the precise location metrics from data/Variable_data_MiniWorld.
Uses local google/gemma-4-12B-it on GPU.
"""

import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
from omegaconf import DictConfig

# Allow importing project modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

SYSTEM_PROMPT = """You are an expert descriptive annotator for 3D reinforcement learning environments.
Your task is to produce a natural, richly phrased scene caption by synthesizing:
1. The linguistic variability, varied sentence structure, and scene context from the Reference Phrasing.
2. The PRECISE numerical location measurements (distance in units, angle in degrees, and direction) from the Precise Ground Truth.

CRITICAL RULES:
1. STRICT NUMERICAL FIDELITY: When objects exist, you MUST preserve every exact distance (e.g., "1.2 units", "4.5 units") and angle (e.g., "24.0 degrees to the left", "0.2 degrees to the left") verbatim from the Precise Ground Truth. Never alter, round, or omit any numerical value or direction.
2. NO HALLUCINATED OBJECTS: MiniWorld scenes contain ONLY boxes and balls. NEVER hallucinate, invent, or mention non-existent objects like chairs, wooden chairs, tables, doors, furniture, or people. Only describe objects that are explicitly listed in the Precise Ground Truth.
3. EMPTY SCENES (NO HALLUCINATED MEASUREMENTS): If the Precise Ground Truth indicates no objects are present, describe ONLY the empty room elements (green grass floor, gray walls, blue sky). NEVER invent units or degrees in empty scenes, and NEVER attach measurements (e.g., "1.2 units to the left") to walls, floors, or empty space.
4. NATURAL LINGUISTIC VARIABILITY: Do NOT copy the repetitive robotic formula ("A [object] is visible at X units and Y degrees to the left, located on a green grass floor surrounded by gray walls under a blue sky."). Instead, adopt varied sentence openings, diverse descriptions of the floor, walls, and sky, and natural spatial phrasing inspired by the Reference Phrasing.
5. NO PROPER NAMES OR PERSONIFIED STYLES: Never use character names, artist styles, or personifications (e.g., do NOT use "Nuss", "Williams-style", "Johnson's room", "Johnson-colored").
6. COMPLETE SENTENCES & CLEAN SYNTAX: Every sentence must be fully formed and grammatically complete with proper ending punctuation. Never leave truncated clauses or dangling prepositions at the end (e.g., do NOT end with "under a.", "beneath a.", "and.", "with.", "enclosed by.").
7. STRICT OUTPUT FORMAT (NO CHATBOT LEAKS): Output ONLY the final caption text starting directly with a capital letter. NEVER output conversational greetings, acknowledgments, or refusals (e.g., do NOT say "Hi there!", "I'm ready to help", "Please provide the Reference Phrasing..."). Do NOT output bullet points, dashes, leading dots, quotes, or prefixes like "Final Caption:"."""

PROMPT_TEMPLATE = """Reference Phrasing: {noisy}
Precise Ground Truth: {var}

Final Caption:"""

PROMPT_TEMPLATE_NO_REF = """Precise Ground Truth: {var}

Write a natural, varied scene description strictly preserving all exact numerical measurements (distance in units, angle in degrees, direction) and object identities:
Final Caption:"""


def build_prompt(noisy_text: str, var_text: str) -> str:
    if noisy_text and noisy_text.strip():
        content = f"{SYSTEM_PROMPT}\n\n" + PROMPT_TEMPLATE.format(
            noisy=noisy_text.strip(), var=var_text.strip()
        )
    else:
        content = f"{SYSTEM_PROMPT}\n\n" + PROMPT_TEMPLATE_NO_REF.format(
            var=var_text.strip()
        )
    return content


def clean_output(text: str) -> str:
    # Strip turn / stop tokens
    for stop_seq in ["<turn|>", "<|turn>", "<eos>", "<pad>", "<|channel|>"]:
        if stop_seq in text:
            text = text.split(stop_seq)[0]

    # Remove internal channels / thoughts
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel\|>analysis.*?<\|end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel\|>final<\|message\|>", "", text)
    text = re.sub(r"<\|return\|>", "", text)

    for tok in ["<eos>", "<turn|>", "<|turn>", "<pad>"]:
        text = text.replace(tok, "")

    text = text.strip().strip('"\'')
    # Strip common prefixes
    for prefix in [
        "final caption:",
        "caption:",
        "paraphrased caption:",
        "paraphrase:",
        "description:",
        "title:",
        "3d scene:",
    ]:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()

    # Strip any leading single-word label followed by colon (e.g. 'Title:', '제목:')
    text = re.sub(r'^[^\s:]+:\s*', '', text)

    # Take first valid line if multi-line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        text = lines[0]

    text = text.strip().strip('"\'')

    # Strip leading dots, bullets, and dashes
    text = re.sub(r'^\.+[ \t]*', '', text)
    text = re.sub(r'^[\-\*•][ \t]*', '', text)

    # Strip generator artifact prefixes
    text = re.sub(r'^INE\s+', '', text)
    text = re.sub(r'^Ah,\s*', 'A ', text)
    text = re.sub(r'^Alert,\s*', 'The ', text)
    text = re.sub(r'^opnly\s+', 'Only ', text, flags=re.IGNORECASE)
    text = re.sub(r'^opining\s+', 'Opening ', text, flags=re.IGNORECASE)
    text = re.sub(r'^opined\s+', 'Opening ', text, flags=re.IGNORECASE)
    text = re.sub(r'^osite\s+', 'Opposite ', text, flags=re.IGNORECASE)
    text = re.sub(r'^op\s+the\s+horizon', 'On the horizon', text, flags=re.IGNORECASE)

    # Clean hallucinated styles
    text = re.sub(r'\bWilliams-style\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bJohnson-style\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r"\bJohnson's\s+room\b", 'The room', text, flags=re.IGNORECASE)
    text = re.sub(r'\bJohnson-colored\s+walls\b', 'Gray walls', text, flags=re.IGNORECASE)
    text = re.sub(r'\bJohnson-green\s+flooring\b', 'Green flooring', text, flags=re.IGNORECASE)

    # Capitalize first letter if lowercase
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    if text and text[-1] not in ".!?":
        text += "."
    return text


def extract_measurements(text: str) -> List[str]:
    """Extract numeric measurements like '1.2 units' or '24.0 degrees'."""
    matches = re.findall(r"\b\d+\.?\d*\s*(?:units|degrees)", text, flags=re.IGNORECASE)
    return [m.lower().strip() for m in matches]


def validate_numerical_integrity(generated: str, ground_truth: str) -> bool:
    """Verify that generated caption is complete, accurate, and free of hallucinations."""
    if not generated or len(generated.split()) < 4:
        return False

    gen_lower = generated.lower()
    gt_lower = ground_truth.lower()

    # Reject chatbot leaks and prompt repetitions
    if any(k in gen_lower for k in ["ready to help", "reference phrasing", "ground truth", "**reference", "final caption:"]):
        return False

    # Reject hallucinated objects not present in MiniWorld
    if "chair" in gen_lower or "table" in gen_lower or re.search(r"\bperson\b", gen_lower) or re.search(r"\bnuss\b", gen_lower):
        return False

    # Reject dangling truncated endings
    dangling_pat = re.compile(
        r'\b(?:and|a|an|the|with|in|on|at|to|of|under|beneath|from|by|as|that|or|for|is|are|was|were|pe|horizo)\s*[\.\?!]+$',
        re.IGNORECASE
    )
    if dangling_pat.search(generated.strip()):
        return False

    # Check numerical measurements
    gt_measurements = extract_measurements(ground_truth)
    gen_measurements = extract_measurements(generated)

    # In empty scenes (no measurements in ground truth), generated text must NOT have hallucinated measurements
    if not gt_measurements:
        # Ground truth has no measurements
        if gen_measurements and not ("0 units" in gt_lower and "0 units" in gen_lower):
            return False
        return True

    # In scenes with objects, verify all ground truth measurements are preserved
    for m in gt_measurements:
        if m not in gen_lower:
            m_no_space = m.replace(" ", "")
            if m_no_space not in gen_lower:
                return False

    return True


def deterministic_fallback(ground_truth: str, noisy_ref: str) -> str:
    """Deterministic high-quality fallback if model dropped a numerical measurement."""
    # Preserve exact numbers by formatting ground truth cleanly
    text = ground_truth.strip()
    return text


def setup_image_symlink(source_images_dir: Path, output_images_dir: Path, link_mode: str = "symlink"):
    if output_images_dir.is_symlink() or output_images_dir.exists():
        return
    if link_mode == "symlink":
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


def generate_dataset(
    var_dir: str,
    noisy_dir: str,
    output_dir: str,
    model_id: str = "google/gemma-4-12B-it",
    batch_size: int = 64,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_new_tokens: int = 80,
    limit: Optional[int] = None,
    link_mode: str = "symlink",
    log_interval: int = 10,
):
    var_path = Path(var_dir)
    noisy_path = Path(noisy_dir)
    output_path = Path(output_dir)

    var_metadata_file = var_path / "metadata.jsonl"
    noisy_metadata_file = noisy_path / "metadata.jsonl"
    var_images_dir = var_path / "images"

    output_path.mkdir(parents=True, exist_ok=True)
    output_metadata_file = output_path / "metadata.jsonl"
    output_images_dir = output_path / "images"

    if not var_metadata_file.exists():
        raise FileNotFoundError(f"Variable metadata file not found at {var_metadata_file}")

    # 1. Setup image link
    if var_images_dir.exists():
        setup_image_symlink(var_images_dir, output_images_dir, link_mode=link_mode)

    # 2. Index noisy metadata by file_name for fast lookup
    print(f"[INFO] Indexing reference noisy metadata from {noisy_metadata_file}...")
    noisy_index: Dict[str, str] = {}
    if noisy_metadata_file.exists():
        with open(noisy_metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    noisy_index[entry["file_name"]] = entry.get("text", "")
                except json.JSONDecodeError:
                    continue
    print(f"[INFO] Loaded {len(noisy_index)} reference captions from MiniWorldNoisy.")

    # 3. Check already processed entries for seamless resumption
    processed_files: Set[str] = set()
    if output_metadata_file.exists():
        with open(output_metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "file_name" in entry:
                        processed_files.add(entry["file_name"])
                except json.JSONDecodeError:
                    continue
        print(f"[INFO] Found {len(processed_files)} already processed entries in output. Resuming...")

    # 4. Gather pending items
    pending_items: List[Tuple[str, str, str]] = []  # (file_name, var_text, noisy_text)
    with open(var_metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                fname = entry.get("file_name", "")
                if fname not in processed_files:
                    var_text = entry.get("text", "")
                    noisy_text = noisy_index.get(fname, "")
                    pending_items.append((fname, var_text, noisy_text))
                    if limit is not None and len(pending_items) >= limit:
                        break
            except json.JSONDecodeError:
                continue

    print(f"[INFO] Total items remaining to generate: {len(pending_items)}")
    if not pending_items:
        print("[INFO] All items are already generated! Nothing to do.")
        return

    # 5. Load model and tokenizer
    print(f"[INFO] Loading tokenizer and model ({model_id}) on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    print("[INFO] Model loaded successfully onto GPU.")

    # 6. Batched Generation
    num_batches = (len(pending_items) + batch_size - 1) // batch_size
    valid_count = 0
    fallback_count = 0

    with open(output_metadata_file, "a", encoding="utf-8") as out_f:
        pbar = tqdm(total=len(pending_items), desc="Generating MiniWorldVariableNoisy")
        for b_idx in range(num_batches):
            batch = pending_items[b_idx * batch_size : (b_idx + 1) * batch_size]
            formatted_prompts = []

            for fname, var_text, noisy_text in batch:
                p_text = build_prompt(noisy_text, var_text)
                msgs = [{"role": "user", "content": p_text}]
                chat_prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(chat_prompt)

            inputs = tokenizer(
                formatted_prompts, return_tensors="pt", padding=True
            ).to(model.device)

            with torch.no_grad():
                do_sample = temperature > 0.0
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p

                outputs = model.generate(**inputs, **gen_kwargs)

            # Decode and validate
            prompt_len = inputs["input_ids"].shape[1]
            for idx, (fname, var_text, noisy_text) in enumerate(batch):
                gen_tokens = outputs[idx][prompt_len:]
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                cleaned = clean_output(decoded)

                # Validate numerical integrity
                if validate_numerical_integrity(cleaned, var_text):
                    final_text = cleaned
                    valid_count += 1
                else:
                    final_text = deterministic_fallback(var_text, noisy_text)
                    fallback_count += 1

                out_entry = {"file_name": fname, "text": final_text}
                out_f.write(json.dumps(out_entry) + "\n")

            out_f.flush()
            pbar.update(len(batch))

            if (b_idx + 1) % log_interval == 0 or (b_idx + 1) == num_batches:
                pbar.set_postfix(
                    valid=valid_count, fallback=fallback_count
                )

        pbar.close()

    print(
        f"\n[INFO] Generation finished! Saved to {output_metadata_file}\n"
        f"       Total processed: {len(pending_items)} (Valid: {valid_count}, Fallback: {fallback_count})"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MiniWorldVariableNoisy dataset using local Gemma-4-12B-it"
    )
    parser.add_argument("--var_dir", type=str, default="data/Variable_data_MiniWorld")
    parser.add_argument("--noisy_dir", type=str, default="data/MiniWorldNoisy")
    parser.add_argument("--output_dir", type=str, default="data/MiniWorldVariableNoisy")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-12B-it")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    parser.add_argument("--log_interval", type=int, default=10)
    return parser.parse_known_args()


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="generate_miniworld_precise_varied",
)
def hydra_main(cfg: DictConfig):
    var_dir = cfg.paths.get("var_dir", "data/Variable_data_MiniWorld")
    noisy_dir = cfg.paths.get("noisy_dir", "data/MiniWorldNoisy")
    output_dir = cfg.paths.get("output_dir", "data/MiniWorldVariableNoisy")
    link_mode = cfg.paths.get("link_mode", "symlink")

    model_id = cfg.model.get("model_id", "google/gemma-4-12B-it")
    batch_size = cfg.model.get("batch_size", 64)
    temperature = cfg.model.get("temperature", 0.2)
    top_p = cfg.model.get("top_p", 0.95)
    max_new_tokens = cfg.model.get("max_new_tokens", 80)

    limit = cfg.options.get("limit", None)
    log_interval = cfg.options.get("log_interval", 10)

    generate_dataset(
        var_dir=var_dir,
        noisy_dir=noisy_dir,
        output_dir=output_dir,
        model_id=model_id,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        limit=limit,
        link_mode=link_mode,
        log_interval=log_interval,
    )


if __name__ == "__main__":
    has_cli = any(
        arg.startswith("--") and not arg.startswith("--hydra") and not arg.startswith("--cfg")
        for arg in sys.argv[1:]
    )
    if has_cli:
        args, _ = parse_args()
        generate_dataset(
            var_dir=args.var_dir,
            noisy_dir=args.noisy_dir,
            output_dir=args.output_dir,
            model_id=args.model_id,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            link_mode=args.link_mode,
            log_interval=args.log_interval,
        )
    else:
        hydra_main()
