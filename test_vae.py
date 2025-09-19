import os
import torch
import wandb
import hydra
import random
import pickle
import numpy as np
import torchvision.utils as vutils

from tqdm import tqdm
from hydra.utils import instantiate
from accelerate import Accelerator
from architectures.common_utils import *
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from accelerate.utils import ProjectConfiguration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_descriptions_to_disk(descriptions_list, output_directory="result"):
    """
    Saves a nested list of text descriptions into individual .txt files.

    Args:
        descriptions_list (list): A nested list of strings.
        output_directory (str): The name of the directory to save files in.
    """
    # Create the target directory if it doesn't already exist
    os.makedirs(output_directory, exist_ok=True)
    print(f"Directory '{output_directory}' is ready. 📂")

    file_count = 0
    # Iterate through the outer list with an index (e.g., group_idx)
    for group_idx, sublist in enumerate(descriptions_list):
        # Iterate through the inner list of descriptions with an index (e.g., desc_idx)
        for desc_idx, description in enumerate(sublist):
            # Create a unique filename based on the indices
            filename = f"description_{group_idx}_{desc_idx}.txt"
            filepath = os.path.join(output_directory, filename)

            # Open the file in write mode and save the description
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(description)
                file_count += 1
            except IOError as e:
                print(f"Error writing file {filepath}: {e}")

    print(f"\nSuccess! ✨ Saved a total of {file_count} files in the '{output_directory}' directory.")

@hydra.main(version_base=None, config_path="config", config_name="test_vae")
def train(args: DictConfig) -> None:
    cfg = args.models
    # Setup Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.accelerator.project_dir, 
        logging_dir=cfg.accelerator.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        mixed_precision=cfg.accelerator.mixed_precision,
        log_with=None, # Use the conditional logger
        project_config=accelerator_project_config,
    )
    
    # --- 1. Define Model ---
    accelerator.print("Initializing VAE model...")
    print("[INFO] Vision model name: ", cfg.project_name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

    #Load the vision models
    vison_model_dir = cfg.test.model_dir
    if os.path.exists(vison_model_dir + "/config.yaml"):
        vision_model_args =  OmegaConf.load(vison_model_dir + "/config.yaml")
        cfg = vision_model_args.models
        cfg.test.model_dir = vison_model_dir
    else:
        raise FileNotFoundError(f"Config file not found in {vison_model_dir}/config.yaml")
    accelerator.print("Initializing VAE model...")
    vision_model = instantiate(cfg.model)
    vision_model.load_params(f"{vison_model_dir}/{cfg.project_name}.tar")
    
    # --- Prepare the data ---
    env = get_env(args.env)
    data, _ = collect_data(env, args.num_data)
    changed_descriptions = change_descriptions(data)
    save_descriptions_to_disk(changed_descriptions)
    # --- Prepare for Distributed Training ---
    vision_model = accelerator.prepare(vision_model)
    vision_model.eval()
    # --- Testing the model ---
    vision_model.test(data, changed_descriptions, args.result_dir)
    
     
if __name__ == "__main__":
    train()