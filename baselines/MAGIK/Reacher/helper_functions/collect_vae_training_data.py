import sys
sys.path.append('.')

import os
import torch
import pickle
import argparse
import numpy as np
from utils import load_config

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
        default=100,
        help='Number of episodes to run for data collection (default: 100)'
    )

    args = parser.parse_args()
    return args

def select_action_random(action_space):
    """
    Select a random action.
    
    Args:
        action_space: The action space of the environment.
    
    Returns:
        int: Selected action.
    """
    return action_space.sample()

# =============================
# 4. Data Collection
# =============================

def collect_data(env, episodes):
    """
    Collect transition data from the specified environment using the random policy.
    
    Args:
        env_name (str): Name of the MiniGrid environment.
        episodes (int): Number of episodes to run.

    Returns:
        list: Collected state and their class annotations
    """


    for episode in range(1, episodes + 1):
        state, info = env.reset()
        done = False
        step = 0
        while not done:
            action = select_action_random(env.action_space)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated + truncated
            step += 1
            # env.render()
        print(f"Episode {episode}/{episodes} finished after {step} steps.")
    
    data = env.obs_data
    class_prob =  env.obs_class

    env.close()
    counts = {}
    for element in class_prob:
        counts[element] = counts.get(element, 0) + 1

    # Print each element and its count
    for element, count in counts.items():
        print(f"{element}: {count}")

    # Number of unique categories
    num_categories = np.max(class_prob) + 1
    # One-hot encode
    class_prob = np.eye(num_categories)[class_prob]

    return data, class_prob

# =============================
# 5. Main Function
# =============================
def main():
    args = parse_arguments()

    env_name = args.env_name
    episodes = args.episodes
    
    # Create data directory if it doesn't exist
    data_dir = f'data/{env_name}/feature_based/Random'
    os.makedirs(data_dir, exist_ok=True)
    
    if args.env_name == "reacher":
        from env.reacher import Reacher4TargetsEnv
        env_config = load_config("config/reacher.yml")
        env = Reacher4TargetsEnv(goal_target_idx=env_config["goal_target_idx"], max_steps = env_config["max_steps"], reaching_threshold=env_config["reaching_threshold"])

    else:
        raise NotImplementedError(f"Environment {env_name} is not implemented.")
    
    # Collect data
    data, class_prob = collect_data(env=env, episodes=episodes)

    data = np.stack(data)
    
    # Save data
    data_path = os.path.join(data_dir, 'states.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(np.stack(data), f)
    print(f"Collected {len(data)} transitions and saved to {data_path}")
    
    data_path = os.path.join(data_dir, 'class_prob.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(np.stack(class_prob), f)
    print(f"Collected {len(class_prob)} class prob and saved to {data_path}")
# =============================
# 6. Entry Point
# =============================

if __name__ == "__main__":
    main()
