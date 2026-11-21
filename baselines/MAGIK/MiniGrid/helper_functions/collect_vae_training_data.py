import os
import sys
sys.path.append('.')
import hydra
import pickle
import argparse
import minigrid
import gymnasium
import numpy as np

from PIL import Image
from omegaconf import DictConfig, OmegaConf

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect transition data from MiniGrid environment.")
    parser.add_argument(
        '--env',
        type=str,
        default = "SimplePickup",
        help='Name of the MiniGrid environment (e.g., MiniGrid-Empty-5x5-v0)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes to run for data collection (default: 100)'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=20,
        help='Maximum steps per episode (default: 100)'
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

def collect_data(env, episodes):
    """
    Collect transition data from the specified environment using the chosen policy.
    

    Returns:
        list: Collected transitions and classes.
    """
    data = []
    captions = []
    class_prob = []
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        done = False
        step = 0
        while not done:
            action = select_action_random(env.action_space)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated + truncated
            cls = env.get_class(env.obs)
            if isinstance(next_state, dict):
                data.append(np.transpose(next_state['image']/255, (2,0,1)))
            else:
                data.append(np.transpose(next_state/255, (2,0,1)))
            class_prob.append(cls)
            state = next_state
            step += 1
        
        print(f"Episode {episode}/{episodes} finished after {step} steps.")
    
    env.close()
    return data, captions, class_prob

@hydra.main(version_base=None, config_path="../config", config_name="3DRoom")
def main(arg: DictConfig) -> None:
    args = parse_arguments()
    
    env_name = args.env
    episodes = args.episodes
    max_steps = args.max_steps
    

    # Create data directory if it doesn't exist
    data_dir = f'data/{env_name}/Random'
    os.makedirs(data_dir, exist_ok=True)
    
    if env_name == "SimplePickup":
        from env.env import MultiObjectMiniGridEnv #TransitionCaptioner
        env = MultiObjectMiniGridEnv(max_steps=max_steps, agent_view_size=5, size=7, to_be_terminated=True)
        from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    elif env_name == "3DRoom":
        from env.miniworld import ColorBoxesEnv
        env = ColorBoxesEnv(max_episode_steps=max_steps, floor_type='asphalt')
        
    # Collect data
    data, captions, class_prob = collect_data(env=env, episodes=episodes)
    
    # Save data
    data_path = os.path.join(data_dir, 'data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Collected {len(data)} transitions and saved to {data_path}")
    
    data_path = os.path.join(data_dir, 'captions.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(captions, f)
    print(f"Collected {len(captions)} captions and saved to {data_path}")
    
    data_path = os.path.join(data_dir, 'class_prob.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(class_prob, f)
    print(f"Collected {len(class_prob)} class prob and saved to {data_path}")
# =============================
# 6. Entry Point
# =============================

if __name__ == "__main__":
    main()
