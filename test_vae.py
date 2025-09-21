import os
import torch
import wandb
import hydra
import random
import pickle
import numpy as np
import torchvision.utils as vutils

from tqdm import tqdm
from wcwidth import wcswidth
from hydra.utils import instantiate
from accelerate import Accelerator
from architectures.common_utils import *
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from accelerate.utils import ProjectConfiguration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_descriptions_to_disk(descriptions_list, output_directory="result"):
    """
    Processes descriptions and saves them as a visually aligned grid,
    ensuring header and data columns have the same width.
    """

    COLOUR_MAP = {
        'grey': '⚫', 'white': '⚪', 'red': '🔴', 'blue': '🔵',
        'yellow': '🟡', 'green': '🟢', 'brown': '🟤', 'N/A': '➖'
    }
    DEFAULT_EMOJI = '❓'

    # --- Helper function for correct visual padding ---
    def pad_center(text, width):
        """Pads text to a specific visual width."""
        visual_width = wcswidth(text)
        if visual_width >= width:
            return text
        padding = width - visual_width
        left_pad = padding // 2
        right_pad = padding - left_pad
        return (' ' * left_pad) + text + (' ' * right_pad)

    # --- Data Processing (no changes needed here) ---
    processed_data = []
    max_cols = 0
    for sublist in descriptions_list:
        row_data = []
        max_cols = max(max_cols, len(sublist))
        for description in sublist:
            wall_match = re.search(r'(\w+) walls', description)
            wall_colour = wall_match.group(1) if wall_match else "N/A"
            ball_colours = re.findall(r'a (\w+) ball', description)
            ball1_colour = ball_colours[0] if len(ball_colours) > 0 else "N/A"
            ball2_colour = ball_colours[1] if len(ball_colours) > 1 else "N/A"
            
            emojis = (
                COLOUR_MAP.get(wall_colour, DEFAULT_EMOJI),
                COLOUR_MAP.get(ball1_colour, DEFAULT_EMOJI),
                COLOUR_MAP.get(ball2_colour, DEFAULT_EMOJI)
            )
            row_data.append(emojis)
        processed_data.append(row_data)
    
    # --- File Writing with Consistent Padding ---
    os.makedirs(output_directory, exist_ok=True)
    output_filepath = os.path.join(output_directory, "descriptions_emoji_grid.txt")

    # Define a single visual width for all cells
    cell_width = 12

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # Header Row: Use the helper function for padding
            header_cells = [pad_center(f"j={j}", cell_width) for j in range(max_cols)]
            f.write("".ljust(6) + "|" + "|".join(header_cells) + "|\n")

            # Separator Row: Matches the visual width
            separator = "+".join(["-" * cell_width for _ in range(max_cols)])
            f.write("".ljust(5, '-') + "+" + separator + "+\n")

            # Data Rows
            for i, row in enumerate(processed_data):
                row_header = f"i={i}".ljust(5)
                data_cells = []
                for j in range(max_cols):
                    if j < len(row):
                        wall, ball1, ball2 = row[j]
                        content = f"{wall} {ball1} {ball2}"
                        # Data Cells: Use the same helper function for padding
                        padded_content = pad_center(content, cell_width)
                        data_cells.append(padded_content)
                    else:
                        data_cells.append(" " * cell_width) # Empty cell
                
                f.write(f"{row_header} |" + "|".join(data_cells) + "|\n")

    except IOError as e:
        print(f"Error writing to file {output_filepath}: {e}")
        return

    print(f"✅ Success! Saved the descriptions as an emoji grid to '{output_filepath}'")

    
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