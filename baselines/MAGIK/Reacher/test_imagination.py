import os
import cv2
import hydra
import torch
import argparse
import itertools
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from omegaconf import DictConfig
from stable_baselines3 import SAC
from sklearn.metrics import classification_report
from helper_functions.utils import load_config, write_video
from architectures.m2_vae.dgm import DeepGenerativeModel
from helper_functions.annotate_data import closest_point_index

is_agent = False
is_permute = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

target_mapping = {"Red" : 0, "Green" : 1, "Blue" : 2, "Yellow" : 3}
# Inverting the dictionary
inverse_target_mapping = {v: k for k, v in target_mapping.items()}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect transition data from MiniGrid environment.")
    parser.add_argument(
        '--env_name',
        type=str,
        default = "reacher",
        help='Name of the environment (e.g., MiniGrid-Empty-5x5-v0)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of episodes to run for data collection (default: 100)'
    )

    parser.add_argument(
        '--target',
        type=str,
        default="Yellow",
        help='Which target the agent has to reach'
    )

    args = parser.parse_args()
    return args

def permute_observation(obs):
    """
    Given an observation vector, returns a list of all permuted observations
    where only the target positions and distances are swapped.
    
    Args:
        obs (np.ndarray): Original observation (shape: 20,).
        
    Returns:
        List[np.ndarray]: List of permuted observations.
    """
    # Ensure input is a numpy array
    obs = np.array(obs)
    
    # Extract fixed values (indices that should not change)
    fixed_values = np.concatenate([obs[:4], obs[12:16]])  # cos/sin(theta), angular velocity, fingertip pos
    
    # Extract values to be permuted
    target_positions = obs[4:12].reshape(4, 2)  # 4 target points (x, y)
    distances = obs[16:20]  # Distance to the 4 targets

    # Generate all 4! permutations of indices [0,1,2,3]
    all_permutations = list(itertools.permutations(range(4)))

    permuted_observations = []
    
    for perm in all_permutations:
        # Apply the permutation to target positions and distances
        permuted_target_pos = target_positions[list(perm)].flatten()
        permuted_distances = distances[list(perm)]

        # Construct the new observation
        permuted_obs = np.concatenate([fixed_values[:4], permuted_target_pos, fixed_values[4:], permuted_distances])
        
        permuted_observations.append(permuted_obs)

    return permuted_observations

@hydra.main(version_base=None, config_path="config", config_name="m2")
def main(arg: DictConfig) -> None:
    args = parse_arguments()
    # Load the trained SAC model (adjust the path to your model file)
    sac = SAC.load(arg.Test.agent_checkpoint)
    env_name = args.env_name

    target_to_reach = target_mapping.get(args.target)

    print("[INFO] SAC model loaded from: ", arg.Test.agent_checkpoint)
    print("[INFO] Is agent: ", is_agent)
    if args.target == "Red":
        print("[INFO] Target to reach: 🔴")
    elif args.target == "Yellow":
        print("[INFO] Target to reach: 🟡")
    elif args.target == "Green":
        print("[INFO] Target to reach: 🟢")

    if args.env_name == "reacher":
        from env.reacher import Reacher4TargetsEnv
        env_config = load_config("config/reacher.yml")
        env = Reacher4TargetsEnv(goal_target_idx=target_to_reach, max_steps = env_config["max_steps"], reaching_threshold = 0.10, is_test=True)

    elif args.env_name == "simple_reacher":
        from env.simple_recher import Reacher4TargetsEnv
        env_config = load_config("config/reacher.yml")
        env = Reacher4TargetsEnv(goal_target_idx=env_config["goal_target_idx"], reaching_threshold = env_config.get("reaching_threshold"), max_steps = env_config["max_steps"])
        
    elif args.env_name == "reacherv5":
        env = gym.make("Reacher-v5")

    model = DeepGenerativeModel(arg.Network, torch.tensor(np.ones((4,)))).to(device)
    model.load(arg.Test.vae_checkpoint)
    model.to(device)
    model.eval()

    # Create data directory if it doesn't exist
    if is_agent:
        video_dir = f"Videos/{env_name}/SAC"
    else:
        video_dir = f"Videos/{env_name}/Imagination"

    data = []
    step = 0
    success = 0
    true_classes = []
    predicted_classes = []
    all_misclassified = []
    for i in range(args.episodes):
        state, info = env.reset()
        # Predict an action given the observation
        done = False
        frame_array = []
        imagined_frame_array = []
        while not done:
            if is_agent:
                imagination = state
            else:
                tensor = torch.tensor(state).unsqueeze(0).to(device).float()
                c_h = model.c_encoder(tensor)
                obs_class = torch.argmax(F.softmax(model.classify(c_h), dim=-1))
                true_class = env.get_class()[0]
                true_classes.append(true_class)
                predicted_classes.append(obs_class.item())
                if true_class != obs_class.item():
                    all_misclassified.append([env.get_frame(40), inverse_target_mapping.get(obs_class.item()), inverse_target_mapping.get(true_class)])
                if obs_class.item() == target_to_reach:
                    if is_permute:
                        permut_obs = permute_observation(state)
                        permut_obs = torch.tensor(permut_obs).to(device)
                        c_h = model.c_encoder(permut_obs.float())
                        permut_obs_class = model.classify(c_h)
                        prob_clossness = F.cross_entropy(F.softmax(permut_obs_class, dim=-1), torch.full((24,), 2).to(device), reduction='none')
                        clossest_indices = torch.topk(prob_clossness, 6, largest=False).indices
                        possible_permutations = permut_obs[clossest_indices]
                        dis_clossness = torch.sum(F.mse_loss(possible_permutations, tensor, reduction='none'), dim=-1)
                        imagination = possible_permutations[torch.argmin(dis_clossness)]
                        frame = env.get_frame(feature_obs = imagination.detach().cpu().numpy())
                    else:
                        imagined_state = model.generate(tensor, F.one_hot(torch.tensor(env_config["goal_target_idx"]), 4).to(device).float()) 
                        imagined_state[:,-4:] = torch.round(imagined_state[:,-4:])
                        imagination = imagined_state
                    # imagination[-4:] = imagined_state[0][-4:].detach().cpu().numpy()
                    # imagination[4:12] = imagined_state[0][4:12].detach().cpu().numpy()
                    # plot_two_coordinate_sets(imagined_state[0][4:12].detach().cpu().numpy().reshape(4,2), state[4:12].reshape(4,2), dir="visualisations")
                elif obs_class.item() == target_mapping.get("Blue"):
                    if is_permute:
                        permut_obs = permute_observation(state)
                        permut_obs = torch.tensor(permut_obs).to(device)
                        c_h = model.c_encoder(permut_obs.float())
                        permut_obs_class = model.classify(c_h)
                        prob_clossness = F.cross_entropy(F.softmax(permut_obs_class, dim=-1), torch.full((24,), target_mapping.get(args.target)).to(device), reduction='none')
                        clossest_indices = torch.topk(prob_clossness, 6, largest=False).indices
                        possible_permutations = permut_obs[clossest_indices]
                        dis_clossness = torch.sum(F.mse_loss(possible_permutations, tensor, reduction='none'), dim=-1)
                        imagination = possible_permutations[torch.argmin(dis_clossness)]
                        frame = env.get_frame(feature_obs = imagination.detach().cpu().numpy())
                    else:
                        imagined_state = model.generate(tensor, F.one_hot(torch.tensor(target_mapping.get(args.target)), 4).to(device).float())
                        imagined_state[:,-4:] = torch.round(imagined_state[:,-4:])
                        imagination = imagined_state
                    # imagination[-4:] = imagined_state[0][-4:].detach().cpu().numpy()
                    # imagination[4:12] = imagined_state[0][4:12].detach().cpu().numpy()
                    # plot_two_coordinate_sets(imagined_state[0][4:12].detach().cpu().numpy().reshape(4,2), state[4:12].reshape(4,2), dir="visualisations")
                else:
                    # if is_permute:
                    #     #First change the observation to Blue class
                    #     permut_obs = permute_observation(state)
                    #     permut_obs = torch.tensor(permut_obs).to(device)
                    #     c_h = model.c_encoder(permut_obs.float())
                    #     permut_obs_class = model.classify(c_h)
                    #     prob_clossness = F.cross_entropy(F.softmax(permut_obs_class, dim=-1), torch.full((24,), 2).to(device), reduction='none')
                    #     clossest_indices = torch.topk(prob_clossness, 6, largest=False).indices
                    #     possible_permutations = permut_obs[clossest_indices]
                    #     dis_clossness = torch.sum(F.mse_loss(possible_permutations, tensor, reduction='none'), dim=-1)
                    #     imagination = possible_permutations[torch.argmin(dis_clossness)]
                    #     frame = env.get_frame(feature_obs = imagination.detach().cpu().numpy())
                    #     #Then change to the target class
                    #     permut_obs = permute_observation(imagination.detach().cpu().numpy())
                    #     permut_obs = torch.tensor(permut_obs).to(device)
                    #     c_h = model.c_encoder(permut_obs.float())
                    #     permut_obs_class = model.classify(c_h)
                    #     prob_clossness = F.cross_entropy(F.softmax(permut_obs_class, dim=-1), torch.full((24,), target_mapping.get(args.target)).to(device), reduction='none')
                    #     clossest_indices = torch.topk(prob_clossness, 6, largest=False).indices
                    #     possible_permutations = permut_obs[clossest_indices]
                    #     dis_clossness = torch.sum(F.mse_loss(possible_permutations, tensor, reduction='none'), dim=-1)
                    #     imagination = possible_permutations[torch.argmin(dis_clossness)]
                    #     frame = env.get_frame(feature_obs = imagination.detach().cpu().numpy())
                    # else:
                    #     #First change the observation to Blue class
                    #     imagined_state = model.generate(tensor, F.one_hot(torch.tensor(env_config["goal_target_idx"]), 4).to(device).float())
                    #     #Then change to the target class
                    #     imagined_state = model.generate(imagined_state, F.one_hot(torch.tensor(target_mapping.get(args.target)), 4).to(device).float())
                    #     imagination = imagined_state
                    imagination = tensor
                if isinstance(imagination, torch.Tensor):
                    imagination = imagination.squeeze(0).detach().cpu().numpy()
            frame_array.append(cv2.cvtColor(env.get_frame(), cv2.COLOR_BGR2RGB))
            # concatenated_image = np.concatenate((env.get_frame(features=imagination), env.get_frame()), axis=1)
            # imagined_frame_array.append(cv2.cvtColor(concatenated_image, cv2.COLOR_BGR2RGB))
            action, _states = sac.predict(imagination, deterministic=True)
            # action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            if env.is_reached():
                success += 1
                break
            done = terminated + truncated
            state = next_state
            step += 1 
        write_video(frame_array, f'{str(i)}_full_view', f'{video_dir}/Full_view', (512,512))
    # counts = {}
    # for element in data:
    #     counts[element] = counts.get(element, 0) + 1

    # # Print each element and its count
    # for element, count in counts.items():
    #     print(f"{element}: {count}")

    print(f"Success: {success}/{args.episodes}")
    if not is_agent:
        # Print classification summary after all episodes
        # print("\nClassification Summary:")
        # print(classification_report(true_classes, predicted_classes, target_names=target_mapping.keys()))

        visualisation_dir = 'visualisations'
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

