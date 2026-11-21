import sys
sys.path.append('.')

import os
import wandb
import pickle
import numpy as np
import warnings
import gymnasium as gym
from stable_baselines3 import SAC
from argparse import ArgumentParser
from configparser import ConfigParser
from helper_functions.utils import load_config, create_dump_directory
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

model_path = "model_weights/reacher/SAC/2025-04-05_12-43-17_ZKJBJH/sac_agent"

class WandbLoggingCallback(BaseCallback):
    def __init__(self, episode_length, verbose=1):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.episode_length = episode_length
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self):
        # Accumulate rewards
        reward = self.locals["rewards"][0] if "rewards" in self.locals else 0
        self.episode_rewards.append(reward)

        # Log at truncation (every episode_length steps)
        if self.num_timesteps % self.episode_length == 0:
            episode_reward = np.sum(self.episode_rewards)
            self.episode_count += 1
            log_data = {
                "env_step": self.num_timesteps,
                "episode_reward": episode_reward,
                "episode": self.episode_count
            }
            wandb.log(log_data, step=self.num_timesteps)
            self.episode_rewards = []  # Reset for next episode

        return True

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'reacher', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--timestep", type=int, default = 500_000, help = "Timestep the agent interact with the env")
parser.add_argument("--save_freq", type=int, default = 25_000, help = "Save freequency of the model")
parser.add_argument("--is_fine_tune", type=bool, default = False, help = "Whether to fine tune the model")
parser.add_argument("--target", type=int, default = 3, help = "Target to reach")
args = parser.parse_args()
parser = ConfigParser()

# Create the Reacher environment
if args.env_name == "reacher":
    from env.reacher import Reacher4TargetsEnv
    env_config = load_config("config/reacher.yml")
    if not args.is_fine_tune:
        target = env_config["goal_target_idx"]
    else:
        target = args.target
    env = Reacher4TargetsEnv(goal_target_idx=target, max_steps = env_config["max_steps"])
elif args.env_name == "reacherv5":
    env = gym.make("Reacher-v5")

# Initialize wandb
wandb.init(project="SAC_Mujoco", name=f"{target}_fine_tune" if args.is_fine_tune else f"{target}_train", config=args, reinit=True)

model_save_dir = f"model_weights/{args.env_name}/SAC"
model_save_dir = create_dump_directory(model_save_dir)
print("[INFO] Model save directory: ", model_save_dir)

if args.is_fine_tune:
    learning_starts = 0
else:
    learning_starts = 5_000

# Instantiate the SAC agent
model = SAC("MlpPolicy", env, verbose=0, buffer_size=int(args.timestep/20), learning_starts=learning_starts)

if env_config["goal_target_idx"] != 2 and not args.is_fine_tune:
    warnings.warn(f"Goal target is not 2 ({env_config['goal_target_idx']}) and fine tune is disabled. Do you intent to fine tune?")

if args.is_fine_tune:
    model = SAC.load(model_path)
    print("[INFO] Fine-Tuning model loaded from: ", model_path)
    model.set_env(env)

# Log model parameters and gradients using wandb.watch()
wandb.watch(model.policy, log="all", log_freq=100)
wandb.config.update(args)

# Train the agent
timesteps = args.timestep

checkpoint_callback = CheckpointCallback(
    save_freq=args.save_freq,  # Save every 10,000 steps
    save_path=model_save_dir,
    name_prefix="sac_reacher"
)
model.learn(total_timesteps=timesteps, callback=[WandbLoggingCallback(episode_length=50), checkpoint_callback])

# Save the trained model
model.save(f"{model_save_dir}/sac_agent")
data = env.obs_data
class_prob =  env.obs_class

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
data_path = "data/reacher/feature_based/SAC"
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