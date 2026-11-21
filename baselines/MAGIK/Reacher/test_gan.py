import os
import cv2
import hydra
import torch
import random
import warnings
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from omegaconf import DictConfig
from stable_baselines3 import SAC
from architectures.gan.gan import SSGAN
from torchvision.utils import save_image
from sklearn.metrics import classification_report
from helper_functions.visualize import Visualizer
from helper_functions.utils import write_video, load_config
from helper_functions.annotate_data import closest_point_index

is_agent = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore")

target_mapping = {"Red" : 0, "Green" : 1, "Blue" : 2, "Yellow" : 3}
# Inverting the dictionary
inverse_target_mapping = {v: k for k, v in target_mapping.items()}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect transition data from MiniGrid environment.")
    parser.add_argument(
        '--env',
        type=str,
        default='reacher',
        help='Which env to test'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of episodes to run for data collection (default: 200)'
    )

    parser.add_argument(
        '--target',
        type=str,
        default="Blue",
        help='Which target the agent has to reach'
    )

    args = parser.parse_args()
    return args

@hydra.main(version_base=None, config_path="config", config_name="gan")
def main(arg: DictConfig) -> None:
    args = parse_arguments()
    # Load the trained SAC model (adjust the path to your model file)
    sac = SAC.load(arg.Test.agent_checkpoint)
    print("[INFO] SAC model loaded from: ", arg.Test.agent_checkpoint)
    print("[INFO] Is agent: ", is_agent)
    if args.target == "Red":
        print("[INFO] Target to reach: 🔴")
    elif args.target == "Yellow":
        print("[INFO] Target to reach: 🟡")
    elif args.target == "Green":
        print("[INFO] Target to reach: 🟢")
    
    target_to_reach = target_mapping.get(args.target)

    env_name = args.env

    if env_name == "reacher":
        from env.reacher import Reacher4TargetsEnv
        env_config = load_config("config/reacher.yml")
        env = Reacher4TargetsEnv(goal_target_idx=target_to_reach, reaching_threshold = env_config.get("reaching_threshold"), \
                                 max_steps = env_config["max_steps"])
    
    elif env_name == "simple_reacher":
        from env.simple_recher import Reacher4TargetsEnv
        env_config = load_config("config/reacher.yml")
        env = Reacher4TargetsEnv(goal_target_idx=target_to_reach, reaching_threshold = env_config.get("reaching_threshold"), \
                                 max_steps = env_config["max_steps"], reaching_patience = env_config["reaching_patience"])
        
    elif env_name == "two_target_reacher":
        from env.two_target_reacher import Reacher2TargetsEnv
        env_config = load_config("config/reacher.yml")
        env = Reacher2TargetsEnv(goal_target_idx=target_to_reach, reaching_threshold = env_config.get("reaching_threshold"), \
                                 max_steps = env_config["max_steps"], reaching_patience = env_config["reaching_patience"])
        
    elif env_name == "three_target_reacher":
        from env.three_target_reacher import Reacher3TargetsEnv
        env_config = load_config("config/reacher.yml")
        env = Reacher3TargetsEnv(goal_target_idx=target_to_reach, reaching_threshold = env_config.get("reaching_threshold"), \
                                 max_steps = env_config["max_steps"], reaching_patience = env_config["reaching_patience"])
        
    elif env_name == "reacherv5":
        env = gym.make("Reacher-v5")
        
    if not is_agent:
        model = SSGAN(arg.Network, 10, device).to(device)
        model.load(arg.Test.gan_checkpoint)
        model.to(device)
        model.eval()

    # Create data directory if it doesn't exist
    if is_agent:
        video_dir = f"Videos/{env_name}/SAC"
    else:
        video_dir = f"Videos/{env_name}/Imagination"

    all_misclassified = []
    true_classes = []
    predicted_classes = []
    step = 0
    success = 0
    for i in range(args.episodes):
        state, info = env.reset()
        done = False
        frame_array = []
        while not done:
            if is_agent:
                imagination = state
            else:
                tensor = torch.tensor(state).unsqueeze(0).to(device).float()
                c = model.classify(tensor)
                obs_class = torch.argmax(F.softmax(c, dim=-1))
                true_class = env.get_class()[0]
                true_classes.append(true_class)
                predicted_classes.append(obs_class.item())

                if obs_class.item() != true_class:
                    all_misclassified.append((env.get_frame(40), obs_class, true_class))
                if obs_class == target_to_reach:
                    imagination = model.generate(tensor, F.one_hot(torch.tensor(env_config["goal_target_idx"]), 4).to(device).float())
                    imagination[:,-4:] = torch.round(imagination[:,-4:])
                elif obs_class == env_config["goal_target_idx"]:
                    imagination = model.generate(tensor, F.one_hot(torch.tensor(target_mapping.get(args.target)), 4).to(device).float())
                    imagination[:,-4:] = torch.round(imagination[:,-4:])
                else:
                    imagination = state
                if isinstance(imagination, torch.Tensor):
                    imagination = imagination.squeeze(0).detach().cpu().numpy()
            frame_array.append(cv2.cvtColor(env.get_frame(), cv2.COLOR_BGR2RGB))
            action, _states = sac.predict(imagination, deterministic=True)
            # action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated + truncated
            if env.is_reached():
                success += 1
                break
            state = next_state
            step += 1  
        write_video(frame_array, f'{str(i)}_full_view', f'{video_dir}/Full_view', (512,512))
       
    
    print(f"Success: {success}/{args.episodes}")
    if not is_agent:
        # Print classification summary after all episodes
        print("\nClassification Summary:")
        print(classification_report(true_classes, predicted_classes, target_names=target_mapping.keys()))

    if not is_agent:
        visualisation_dir = 'visualizations'
        os.makedirs(visualisation_dir, exist_ok=True)

        # === Display Misclassified Images in a Grid ===
        # Display up to 25 misclassified images
        n_display = min(25, len(all_misclassified))
        if n_display > 0:
            n_cols = 5
            n_rows = (n_display + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
            axes = axes.flatten()
            for i in range(n_display):
                image, pred, true_label = all_misclassified[i]
                # If image is channel-first (C, H, W) then convert to (H, W, C)
                if image.shape[0] == 3 or image.shape[0] == 1:
                    image_disp = np.transpose(image, (1, 2, 0))
                else:
                    image_disp = image
                axes[i].imshow(image_disp)
                axes[i].axis('off')
                axes[i].set_title(f"Pred: {pred}\nTrue: {true_label}", fontsize=10)
            # Turn off any unused subplots
            for j in range(n_display, len(axes)):
                axes[j].axis('off')
            plt.tight_layout()
            plt.savefig(f"{visualisation_dir}/misclassified.png")
        else:
            print("No misclassified samples to display.")


if __name__ == "__main__":
    main()

