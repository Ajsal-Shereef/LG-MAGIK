import os
import json
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

@hydra.main(version_base=None, config_path="config", config_name="text_encoder")
def main(cfg: DictConfig):
    print(f"--- Starting Fine-tuning for {cfg.env.name} ---")
    
    # 1. Load Data
    caption_file = cfg.data.caption_file
    if not os.path.exists(caption_file):
        raise FileNotFoundError(f"Caption file not found at: {caption_file}")
    
    print(f"Loading captions from: {caption_file}")
    train_sentences = []
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "text" in data:
                    train_sentences.append(data["text"])
            except json.JSONDecodeError:
                continue
                
    print(f"Found {len(train_sentences)} captions.")
    if len(train_sentences) == 0:
        raise ValueError("No captions found in the dataset file.")

    # 2. Prepare Training Data for Unsupervised SimCSE
    # We create InputExample with [text, text]. MultipleNegativesRankingLoss 
    # will treat them as a positive pair. The dropout in the model acts as noise.
    train_examples = [InputExample(texts=[sent, sent]) for sent in train_sentences]
    
    # DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=cfg.training.batch_size)
    
    # 3. Initialize Model
    print(f"Initializing model: {cfg.training.model_name}")
    
    if "openai/clip" in cfg.training.model_name.lower():
        from sentence_transformers import models
        from transformers import CLIPConfig, CLIPTextModel, CLIPTokenizer
        
        # 1. Word Embedding Model (Text Tower)
        try:
             # Load from HF directly
             clip_text = CLIPTextModel.from_pretrained(cfg.training.model_name)
             clip_tokenizer = CLIPTokenizer.from_pretrained(cfg.training.model_name)
        except Exception as e:
             raise RuntimeError(f"Failed to load CLIPTextModel from {cfg.training.model_name}: {e}")
             
        word_embedding_model = models.Transformer(model_name_or_path=cfg.training.model_name)
        word_embedding_model.auto_model = clip_text
        word_embedding_model.tokenizer = clip_tokenizer
        
        # 2. Pooling
        dim = clip_text.config.hidden_size
        pooling_model = models.Pooling(dim, pooling_mode_mean_tokens=True)
        
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(cfg.training.model_name)
    
    # 4. Loss Function
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 5. WandB Setup
    if cfg.wandb.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Callback to log loss
    def callback(score, epoch, steps):
        if cfg.wandb.use_wandb:
            wandb.log({"loss": score, "epoch": epoch, "step": steps})

    # 5. Training
    output_path = cfg.training.output_dir
    checkpoint_path = os.path.join(output_path, "checkpoints")
    steps_per_epoch = len(train_dataloader)
    
    print(f"Starting training for {cfg.training.num_epochs} epochs...")
    print(f"Output directory: {output_path}")
    print(f"Checkpoint directory: {checkpoint_path}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=cfg.training.num_epochs,
        optimizer_params={'lr': cfg.training.learning_rate},
        output_path=output_path,
        show_progress_bar=True,
        callback=callback,
        checkpoint_path=checkpoint_path,
        checkpoint_save_steps=steps_per_epoch,
        checkpoint_save_total_limit=2 # Keep last 2 epochs to avoid filling disk
    )
    
    if cfg.wandb.use_wandb:
        wandb.finish()
    print("--- Training Completed ---")
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()