import os
from PIL import Image
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from datasets import load_dataset
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
from architectures.common_utils import create_dump_directory

# Custom Trainer to fix 'num_items_in_batch' argument mismatch
class CustomBlipTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs) 
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# --- Main function decorated with Hydra ---
@hydra.main(config_path="config", config_name="train_captioner", version_base=None)
def main(cfg: DictConfig):
    
    # --- Get Original Working Directory ---
    # Hydra changes the working dir, so we use get_original_cwd() 
    # to build absolute paths to our data/models.
    original_cwd = get_original_cwd()
    
    # --- TRAIN MODE ---
    if cfg.mode == 'train':
        # --- Define Paths from Config ---
        dataset_path = os.path.join(original_cwd, cfg.paths.dataset_path)
        model_save_dir = create_dump_directory(cfg.paths.final_model_path)
        final_model_path = f"{model_save_dir}/final"
        os.makedirs(final_model_path, exist_ok=True)
        
        print(f"Running in TRAIN mode using base model: {cfg.model.base_model_id}")
        print(f"Full config:\n{OmegaConf.to_yaml(cfg)}")

        # --- 1. SETUP: Load the dataset ---
        dataset = load_dataset("imagefolder", data_dir=dataset_path)
        print("Dataset loaded:", dataset)

        # --- 2. INITIALIZATION: Load BLIP processor and model ---
        processor = BlipProcessor.from_pretrained(cfg.model.base_model_id, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(cfg.model.base_model_id)
        print("Model and processor loaded.")

        # --- 3. PREPROCESSING ---
        def preprocess_function(examples):
            images = [img.convert("RGB") for img in examples["image"]]
            texts = examples["text"]
            
            inputs = processor(
                images=images, 
                text=texts, 
                padding="max_length", 
                max_length=cfg.preprocessing.max_length, 
                truncation=True,
                return_tensors="pt"
            )
            inputs['labels'] = inputs['input_ids']
            return inputs

        print("Mapping preprocessing function...")
        processed_dataset = dataset.map(
            function=preprocess_function, 
            batched=True, 
            remove_columns=dataset["train"].column_names
        )
        
        # --- 4. CALCULATE STEPS ---
        per_device_batch_size = cfg.training.params.per_device_train_batch_size
        train_dataset_size = len(processed_dataset["train"])
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        total_train_batch_size = per_device_batch_size * n_gpus
        
        steps_per_epoch = train_dataset_size // total_train_batch_size
        if train_dataset_size % total_train_batch_size != 0:
            steps_per_epoch += 1

        save_and_eval_steps = steps_per_epoch * cfg.training.eval_every_n_epochs
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Saving & Evaluating every {cfg.training.eval_every_n_epochs} epochs ({save_and_eval_steps} steps)")

        # --- 5. TRAINING ---
        # Convert Hydra config to a standard dict for Seq2SeqTrainingArguments
        training_args_dict = OmegaConf.to_container(cfg.training.params, resolve=True)
        
        # Add dynamic/calculated values
        training_args_dict['fp16'] = torch.cuda.is_available()
        training_args_dict['save_steps'] = save_and_eval_steps
        training_args_dict['eval_steps'] = save_and_eval_steps
        result_save_path = f"{model_save_dir}/result"
        training_args_dict['output_dir'] = result_save_path
        
        training_args = Seq2SeqTrainingArguments(**training_args_dict)
        
        early_stop_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience
        )

        trainer = CustomBlipTrainer(
            model=model,
            args=training_args,
            tokenizer=processor,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["test"],
            data_collator=default_data_collator,
            callbacks=[early_stop_callback],
        )

        print("\nStarting training...")
        print(f"\nResults are saved to: {result_save_path}")
        trainer.train()
        print("Training finished!")
        
        print(f"Saving best model to '{final_model_path}'...")
        trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        print("Model saved.")

    # --- TEST MODE ---
    elif cfg.mode == 'test':
        final_model_path = cfg.inference.test_model_path
        print(f"Running in TEST mode. Loading model from {final_model_path}")

        if not os.path.exists(final_model_path):
            print(f"Error: Model not found at '{final_model_path}'.")
            print("Please run in 'train' mode first.")
            return

        print("Loading model and processor...")
        
        # --- 6. INFERENCE (TESTING) ---
        model = BlipForConditionalGeneration.from_pretrained(final_model_path)
        processor = BlipProcessor.from_pretrained(final_model_path, use_fast=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print("Model loaded successfully.")

        # Load test image paths from config
        test_image_paths = [os.path.join(original_cwd, p) for p in cfg.paths.test_image_paths]
        images = [Image.open(path).convert("RGB") for path in test_image_paths]

        inputs = processor(
            images=images, 
            return_tensors="pt"
        ).to(device)

        # Use generation params from config
        generated_ids = model.generate(
            **inputs,
            max_length=cfg.inference.max_length,
            num_beams=cfg.inference.num_beams,
            early_stopping=cfg.inference.early_stopping
        )
        
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        print("\n--- Generated Captions ---")
        for i, text in enumerate(generated_texts):
            print(f"Image {os.path.basename(test_image_paths[i])}: {text.strip()}")

if __name__ == "__main__":
    main()