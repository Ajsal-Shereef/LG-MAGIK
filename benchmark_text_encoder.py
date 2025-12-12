import pandas as pd
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPModel, CLIPTokenizer

# --- 0. SETUP: FORCE OFFLINE MODE ---
os.environ["HF_HUB_OFFLINE"] = "1"

# --- 1. CONFIGURATION ---
# We separate models by type because they need different loading logic
models_config = [
    # TYPE: 'clip' (Uses transformers library)
    {
        "name": "1. CLIP (Baseline)", 
        "id": "openai/clip-vit-base-patch32", 
        "type": "clip"
    },
    # TYPE: 'st' (Uses sentence-transformers library)
    {
        "name": "2. BGE-Small (Rec)", 
        "id": "BAAI/bge-small-en-v1.5", 
        "type": "st"
    },
    {
        "name": "3. MiniLM (Fastest)", 
        "id": "sentence-transformers/all-MiniLM-L6-v2", 
        "type": "st"
    },
    {
        "name": "4. Snowflake (New)", 
        "id": "Snowflake/snowflake-arctic-embed-xs", 
        "type": "st"
    },
    {
        "name": "6. CLIP (Fine-tuned)", 
        "id": "model_weights/finetuned_clip/MiniWorld/checkpoints/checkpoint-4692", 
        "type": "st"
    },
    {
        "name": "5. MiniLM (Fine-tuned)", 
        "id": "model_weights/finetuned_minilm/MiniWorld", 
        "type": "st"
    }
]

# --- 2. DATA SETUP ---
boilerplate = "The agent is in a room with grass floor. "

anchor_full = "The agent is in a room with grass floor. A blue box is found slightly to the right at angle 16.4."

candidates_full = [
    # 1. Exact Match
    "The agent is in a room with grass floor. A blue box is found slightly to the right at angle 16.4.",
    # 2. Small Numerical Shift (Angle 16.4 -> 17.0)
    "The agent is in a room with grass floor. A blue box is found slightly to the right at angle 17.0.",
    # 3. Different Object
    "The agent is in a room with grass floor. A green ball is found slightly to the right at angle 16.4.",
    # 4. Empty Room
    "The agent is in a room with grass floor. No objects are visible in the current view.",
    # 5. Multiple Objects
    "The agent is in a room with grass floor and concrete walls. A blue box is found to the far left at angle 43.4 at a distance of 0.6 units. A green ball is found in front at angle 1.82 at a distance of 3.1 units."
]

# Helper to clean text
def clean_text(text_list):
    if isinstance(text_list, str):
        return text_list.replace(boilerplate, "")
    return [t.replace(boilerplate, "") for t in text_list]

# --- 3. HELPER: UNIVERSAL MODEL LOADER ---
def get_embeddings(model_config, texts):
    """
    Handles the difference between CLIP (Transformers) and SBERT (SentenceTransformers)
    """
    clean_texts = clean_text(texts)
    
    # CASE A: OpenAI CLIP (Raw Transformers)
    if model_config["type"] == "clip":
        try:
            # Load raw model & tokenizer
            model = CLIPModel.from_pretrained(model_config["id"])
            tokenizer = CLIPTokenizer.from_pretrained(model_config["id"])
            
            # Tokenize and Encode
            inputs = tokenizer(clean_texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                # Get text features (Not normalized by default, but cos_sim handles it)
                embeddings = model.get_text_features(**inputs)
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"CLIP Load Error: {e}")

    # CASE B: Standard Sentence Transformers
    elif model_config["type"] == "st":
        try:
            model = SentenceTransformer(model_config["id"])
            return model.encode(clean_texts, convert_to_tensor=True)
            
        except Exception as e:
            raise RuntimeError(f"ST Load Error: {e}")

# --- 4. COMPARISON LOGIC ---
results = {}

print(f"{'Model':<25} | {'Status'}")
print("-" * 45)

for config in models_config:
    name = config["name"]
    try:
        # Get Anchor Embedding (Shape: 1 x Dim)
        anchor_emb = get_embeddings(config, [anchor_full])
        
        # Get Candidate Embeddings (Shape: N x Dim)
        candidate_embs = get_embeddings(config, candidates_full)
        
        # Compute Cosine Similarity
        # util.cos_sim automatically handles normalization (A . B) / (|A| * |B|)
        scores = util.cos_sim(anchor_emb, candidate_embs)[0].cpu().numpy()
        
        results[name] = scores
        print(f"{name:<25} | Success")
        
    except Exception as e:
        print(f"{name:<25} | FAILED: {str(e)[:40]}...")
        results[name] = [0] * len(candidates_full)

# --- 5. DISPLAY RESULTS ---
print("\n" + "="*80)
print(f"COMPARISON RESULTS (Boilerplate Removed)")
print("="*80)

# Create DataFrame
row_labels = [
    "1. Exact Match", 
    "2. Small Shift (Angle)", 
    "3. Diff Object", 
    "4. Empty Room",
    "5. Multiple Objects"
]
df = pd.DataFrame(results, index=row_labels)

# Calculate Discrimination Power
# (How well does it distinguish 'Small Shift' from 'Different Object'?)
if len(results) >= 2:
    df_gap = pd.DataFrame()
    for col in df.columns:
        # Gap = Similarity(Small Shift) - Similarity(Diff Object)
        # Higher gap = Better at knowing that "Angle Change" is minor but "Object Change" is major
        gap = df.loc["2. Small Shift (Angle)", col] - df.loc["3. Diff Object", col]
        df_gap[col] = [gap]
    
    print("\nDiscrimination Power (Higher is better):")
    print("(Similarity of Small Shift - Similarity of Diff Object)")
    print(df_gap.round(4))

print("\nFull Similarity Table:")
print(df.round(4))