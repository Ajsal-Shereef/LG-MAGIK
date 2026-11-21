
import os
import sys
sys.path.append('.')
import wandb
import pickle
import numpy as np
import warnings
import gymnasium as gym
from stable_baselines3 import SAC, DQN
from argparse import ArgumentParser
from configparser import ConfigParser
from helper_functions.utils import create_dump_directory
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

is_fine_tune = False
model_path = "model_weights/reacher/SAC/2025-03-07_21-35-23_C61GRE/sac_agent"

class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self):
        log_data = {
            "env_step": self.num_timesteps,  # Number of transitions in the environment
            "reward": self.locals["rewards"][0] if "rewards" in self.locals else None,
            "episode_done": self.locals["num_collected_episodes"] if "num_collected_episodes" in self.locals else None
        }
        wandb.log(log_data, step=self.num_timesteps)  # Log with env_step as x-axis
        return True

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = '3DRoom', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--timestep", type=int, default = 1_500_000, help = "Timestep the agent interact with the env")
parser.add_argument("--save_freq", type=int, default = 25_000, help = "Save freequency of the model")
args = parser.parse_args()
parser = ConfigParser()

# Create the Reacher environment
if args.env_name ==  "SimplePickup":
    from env.env import MultiObjectMiniGridEnv, preprocess_observation
    env = MultiObjectMiniGridEnv(max_steps=args.max_ep_len, agent_view_size=5, size=7, \
                                 reward_object=config.env_reward_object, is_single_object = config.is_single_object)
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    exp_name = "SimplePickup"

elif args.env_name == "3DRoom":
    from env.miniworld import ColorBoxesEnv
    env = ColorBoxesEnv(floor_type='asphalt', max_episode_steps=1000)
    exp_name = "3DRoom"

# Initialize wandb
wandb.init(project="SAC_Image", name = exp_name, config=args)

model_save_dir = f"model_weights/{args.env_name}/SAC"
model_save_dir = create_dump_directory(model_save_dir)
print("[INFO] Model save directory: ", model_save_dir)

# Instantiate the SAC agent
model = DQN("CnnPolicy", env, verbose=0, buffer_size=int(args.timestep/20), learning_starts=10000)


if is_fine_tune:
    model = SAC.load(model_path)
    print("[INFO] Fine-Tuning model loaded from: ", model_path)
    model.set_env(env)

# Log model parameters and gradients using wandb.watch()
wandb.watch(model.policy, log_freq=100)
wandb.config.update(args)

# Train the agent
timesteps = args.timestep

checkpoint_callback = CheckpointCallback(
    save_freq=args.save_freq,  # Save every 10,000 steps
    save_path=model_save_dir,
    name_prefix=f"sac_{args.env_name}"
)
model.learn(total_timesteps=timesteps, callback=[WandbLoggingCallback(), checkpoint_callback])

# Save the trained model
model.save(f"{model_save_dir}/sac_agent")
data = env.obs_data
class_prob =  env.obs_class

# counts = {}
# for element in class_prob:
#     counts[element] = counts.get(element, 0) + 1
# # Print each element and its count
# for element, count in counts.items():
#     print(f"{element}: {count}")
# # Number of unique categories
# num_categories = np.max(class_prob) + 1
# # One-hot encode
# class_prob = np.eye(num_categories)[class_prob]
data_path = f"data/{args.env_name}/SAC"
os.makedirs(data_path, exist_ok=True)
# Save data
with open(os.path.join(data_path, 'states.pkl'), 'wb') as f:
    pickle.dump(np.stack(data), f)
print(f"Collected {len(data)} transitions and saved to {data_path}")

with open(os.path.join(data_path, 'class_prob.pkl'), 'wb') as f:
    pickle.dump(np.stack(class_prob), f)
print(f"Collected {len(class_prob)} class prob and saved to {data_path}")

# Close the environment
env.close()

wandb.finish()