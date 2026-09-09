import os
import sys
sys.path.append(os.path.expanduser("~/.local/lib/python3.11/site-packages"))

import json
import random
import argparse
import requests
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install -U sentence-transformers")
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

def get_label(text):
    text = text.lower()
    has_blue = "blue box" in text
    has_green = "green ball" in text
    if has_blue and has_green:
        return "3) Both"
    elif has_green:
        return "2) Green Ball"
    elif has_blue:
        return "1) Blue Box"
    else:
        return "4) Neither"

class SentenceTransformerEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed(self, sentences, batch_size=256):
        return self.model.encode(sentences, batch_size=batch_size, show_progress_bar=True)

class APIEmbedder:
    def __init__(self, api_key, model_name, provider="nvidia"):
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        if provider == "nvidia":
            self.url = "https://integrate.api.nvidia.com/v1/embeddings"
            self.model_name = model_name.replace(":free", "")
        else:
            self.url = "https://openrouter.ai/api/v1/embeddings"
            
    def embed(self, sentences, batch_size=50):
        import time
        all_embeddings = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        dim = 2048
        for i in tqdm(range(0, len(sentences), batch_size), desc=f"{self.provider} API"):
            batch = sentences[i:i+batch_size]
            payload = {
                "input": batch,
                "model": self.model_name,
                "encoding_format": "float"
            }
            if self.provider == "nvidia":
                payload["input_type"] = "query"
                
            max_retries = 10
            for attempt in range(max_retries):
                response = requests.post(self.url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        embeds = [item["embedding"] for item in data["data"]]
                        all_embeddings.extend(embeds)
                        if len(embeds) > 0: dim = len(embeds[0])
                        break
                    else:
                        print("Unexpected response:", data)
                        all_embeddings.extend([np.zeros(dim)] * len(batch))
                        break
                elif response.status_code == 429:
                    time.sleep(2 + attempt)
                else:
                    print(f"API Error {response.status_code}: {response.text}")
                    all_embeddings.extend([np.zeros(dim)] * len(batch))
                    break
            else:
                all_embeddings.extend([np.zeros(dim)] * len(batch))
                
        return np.array(all_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--metadata_path", type=str, default="cached_data/MiniWorld_attempt_7/metadata.jsonl")
    args = parser.parse_args()

    load_dotenv("config/.env")

    print(f"Loading data from {args.metadata_path}...")
    sentences = []
    labels = []
    
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Total entries available: {len(entries)}")
    
    if len(entries) > args.num_samples:
        entries = random.sample(entries, args.num_samples)
        
    for e in entries:
        sentences.append(e["text"])
        labels.append(get_label(e["text"]))

    label_counts = {l: labels.count(l) for l in set(labels)}
    print(f"Label distribution: {label_counts}")

    print("Initializing embedders...")
    embedders = {}
    
    st_models = {
        "1. MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "2. BGE-Large": "BAAI/bge-large-en-v1.5",
        "3. MPNet-Base": "sentence-transformers/all-mpnet-base-v2",
        "4. Para-MPNet": "sentence-transformers/paraphrase-mpnet-base-v2",
        "5. Arctic-XS": "Snowflake/snowflake-arctic-embed-xs",
        "6. BGE-Small": "BAAI/bge-small-en-v1.5"
    }

    try:
        from sentence_transformers import SentenceTransformer
        for name, model_id in st_models.items():
            try:
                embedders[name] = SentenceTransformerEmbedder(model_id)
            except Exception as e:
                print(f"Skipping {name}: {e}")
    except ImportError:
        print("sentence_transformers not found. Skipping ST models.")

    # Nvidia API Embedder
    # api_key = os.getenv("NVIDIA_API_KEY")
    # if api_key:
    #     try:
    #         embedders["1. Llama-Nemotron"] = APIEmbedder(api_key=api_key, model_name="nvidia/llama-nemotron-embed-1b-v2", provider="nvidia")
    #     except Exception as e:
    #         print(f"Could not init Nvidia API: {e}")
    # else:
    #     print("Warning: NVIDIA_API_KEY not found in config/.env! Skipping Nvidia API embedder.")

    out_dir = Path("Embeddings")
    out_dir.mkdir(exist_ok=True)

    # Plot Setup
    sns.set_theme(style="whitegrid")

    for name, embedder in embedders.items():
        print(f"\\n--- Processing {name} ---")
        try:
            emb = embedder.embed(sentences)
            if len(emb) == 0:
                continue
                
            print(f"Shape: {emb.shape}. Running 2D t-SNE for {name}...")
            tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto')
            emb_2d = tsne.fit_transform(emb)
            
            print(f"Plotting 2D for {name}...")
            plt.figure(figsize=(12, 10))
            
            # Ensure consistent color mapping across plots
            hue_order = ["1) Blue Box", "2) Green Ball", "3) Both", "4) Neither"]
            palette = {"1) Blue Box": "blue", "2) Green Ball": "green", "3) Both": "purple", "4) Neither": "gray"}
            
            valid_hues = [l for l in hue_order if l in set(labels)]
            
            sns.scatterplot(
                x=emb_2d[:, 0], 
                y=emb_2d[:, 1],
                hue=labels,
                hue_order=valid_hues,
                palette=palette,
                alpha=0.7,
                s=20
            )
                
            plt.title(f"2D t-SNE Embedding Space: {name}")
            plt.legend(title="Labels")
            plt.tight_layout()
            
            safe_name = name.replace(" ", "_").replace(".", "")
            plt.savefig(out_dir / f"{safe_name}_2d.png", bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved {out_dir}/{safe_name}_2d.png")
        except Exception as e:
            print(f"Failed processing model {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
