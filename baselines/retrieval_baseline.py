import os
import re
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from architectures.common_utils import load_text_encoder_and_tokenizer, tokenize_captions


class CaptionRetriever:
    """
    Vector database retrieval baseline for caption mapping.
    Indexes all unique captions from a training dataset (metadata.jsonl)
    using the VAE's text encoder and FAISS (IndexFlatIP with cosine similarity).
    Given an incoming target caption, it retrieves the nearest source caption.
    """
    def __init__(
        self,
        train_dir: str,
        text_encoder_path: Optional[str] = None,
        tokenizer=None,
        text_encoder=None,
        device: Optional[torch.device] = None,
        caption_column: str = "text",
        cache_dir: Optional[str] = None,
        batch_size: int = 256,
        env_name: Optional[str] = None,
    ):
        self.train_dir = self._resolve_train_dir(train_dir, env_name)
        self.caption_column = caption_column
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or self.train_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.text_encoder_name = text_encoder_path or "default_encoder"

        # Re-use existing tokenizer and encoder if provided (saves GPU VRAM)
        if tokenizer is not None and text_encoder is not None:
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder
        elif text_encoder_path:
            print(f"[INFO] Loading text encoder & tokenizer from: {text_encoder_path}")
            self.tokenizer, self.text_encoder = load_text_encoder_and_tokenizer(text_encoder_path, trust_remote_code=True)
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
        else:
            raise ValueError("Either (tokenizer and text_encoder) or text_encoder_path must be provided.")

        self.captions: List[str] = []
        self.index = None
        self.embeddings: Optional[np.ndarray] = None

        self._build_or_load_index()

    def _resolve_train_dir(self, train_dir: str, env_name: Optional[str] = None) -> str:
        """Resolve any variable placeholders like ${env.name} if present."""
        if "${env.name}" in train_dir and env_name:
            train_dir = train_dir.replace("${env.name}", env_name)
        return os.path.abspath(train_dir)

    def _find_metadata_file(self) -> str:
        candidates = [
            os.path.join(self.train_dir, "metadata.jsonl"),
            os.path.join(self.train_dir, "metadata.jsnl"),
            os.path.join(self.train_dir, "agent", "metadata.jsonl"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(
            f"Could not find metadata.jsonl in {self.train_dir}. Checked: {candidates}"
        )

    def _get_cache_paths(self) -> Tuple[str, str, str]:
        encoder_slug = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.basename(self.text_encoder_name.rstrip("/")))
        index_file = os.path.join(self.cache_dir, f"faiss_index_{encoder_slug}.bin")
        captions_file = os.path.join(self.cache_dir, f"faiss_captions_{encoder_slug}.json")
        embeddings_file = os.path.join(self.cache_dir, f"faiss_embeddings_{encoder_slug}.npy")
        return index_file, captions_file, embeddings_file

    @torch.no_grad()
    def _encode_captions(self, captions: List[str], show_progress: bool = False) -> np.ndarray:
        all_embeddings = []
        iterator = range(0, len(captions), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding captions for FAISS index", total=(len(captions) + self.batch_size - 1) // self.batch_size)

        for i in iterator:
            batch = captions[i : i + self.batch_size]
            input_ids, attention_mask = tokenize_captions(self.tokenizer, batch)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            outputs = self.text_encoder(input_ids, attention_mask=attention_mask, return_dict=True)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                pooled = outputs.text_embeds
            else:
                last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                pooled = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

            normed = F.normalize(pooled, p=2, dim=-1)
            all_embeddings.append(normed.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def _build_or_load_index(self):
        index_file, captions_file, embeddings_file = self._get_cache_paths()

        # Check if cache exists
        if os.path.exists(captions_file) and (os.path.exists(index_file) or os.path.exists(embeddings_file)):
            print(f"[INFO] Loading cached retrieval index from {self.cache_dir}...")
            t0 = time.time()
            with open(captions_file, "r", encoding="utf-8") as f:
                self.captions = json.load(f)

            if HAS_FAISS and os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
            elif os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file)
                if HAS_FAISS:
                    dim = self.embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dim)
                    self.index.add(self.embeddings)
            print(f"[INFO] Retrieval index loaded ({len(self.captions)} unique captions) in {time.time() - t0:.2f}s.")
            return

        # Build index from scratch
        metadata_file = self._find_metadata_file()
        print(f"[INFO] Index cache not found. Building index from {metadata_file}...")
        t0 = time.time()

        # Extract unique captions preserving order of appearance
        seen = set()
        unique_captions = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get(self.caption_column) or data.get("text") or data.get("caption")
                    if text and text not in seen:
                        seen.add(text)
                        unique_captions.append(text)
                except json.JSONDecodeError:
                    continue

        print(f"[INFO] Found {len(unique_captions)} unique captions. Encoding with {self.text_encoder_name}...")
        embeddings = self._encode_captions(unique_captions, show_progress=True)
        self.captions = unique_captions
        self.embeddings = embeddings

        dim = embeddings.shape[1]
        if HAS_FAISS:
            print(f"[INFO] Creating FAISS IndexFlatIP (dim={dim})...")
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            try:
                faiss.write_index(self.index, index_file)
                print(f"[INFO] Saved FAISS index to {index_file}")
            except Exception as e:
                print(f"[WARNING] Could not save FAISS index: {e}")

        # Save embeddings and captions cache
        try:
            np.save(embeddings_file, embeddings)
            with open(captions_file, "w", encoding="utf-8") as f:
                json.dump(unique_captions, f)
            print(f"[INFO] Saved cache to {captions_file} and {embeddings_file}")
        except Exception as e:
            print(f"[WARNING] Could not save captions cache: {e}")

        print(f"[INFO] Index building complete in {time.time() - t0:.2f}s.")

    @torch.no_grad()
    def retrieve(self, target_caption: str, top_k: int = 1) -> Tuple[Union[str, List[str]], Union[float, List[float]]]:
        """
        Retrieves the top-k most similar caption(s) from the source database.
        Returns:
            (top_caption, top_score) if top_k == 1
            ([top_captions], [top_scores]) if top_k > 1
        """
        query_emb = self._encode_captions([target_caption])  # Shape: (1, D)

        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(query_emb, top_k)
            top_indices = indices[0]
            top_scores = scores[0]
        else:
            # Fallback using cosine similarity (inner product on normalized embeddings)
            sims = np.dot(query_emb, self.embeddings.T)[0]
            top_indices = np.argsort(-sims)[:top_k]
            top_scores = sims[top_indices]

        if top_k == 1:
            best_idx = int(top_indices[0])
            best_score = float(top_scores[0])
            return self.captions[best_idx], best_score
        else:
            matched_captions = [self.captions[int(i)] for i in top_indices]
            matched_scores = [float(s) for s in top_scores]
            return matched_captions, matched_scores
