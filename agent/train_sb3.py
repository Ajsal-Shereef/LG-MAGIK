import sys
sys.path.append('.')

import os
import wandb
import pickle
import hydra
import numpy as np
import warnings
import gymnasium as gym
from stable_baselines3 import SAC
from omegaconf import DictConfig, OmegaConf
from architectures.common_utils import create_dump_directory
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

@hydra.main(version_base=None, config_path="../config", config_name="train_agent")
def main(args: DictConfig) -> None:
    # Initialize wandb
    wandb.init(project="LG-MAGIK", name=f"{args.agent.name}_{args.env.name}", config=OmegaConf.to_container(args, resolve=True))

    model_save_dir = create_dump_directory(f"model_weights/{args.agent.name}")
    print("[INFO] Model save directory: ", model_save_dir)
    if args.env.name == "PickEnv":
        from env.PickEnv import PickEnv
        env = PickEnv(args.env)

    # Train the agent
    timesteps = args.env.total_timestep
    # Instantiate the SAC agent
    model = SAC("CnnPolicy", env, verbose=0, buffer_size=int(timesteps/5), learning_starts=15000)

    # Log model parameters and gradients using wandb.watch()
    wandb.watch(model.policy, log="all", log_freq=100)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path=model_save_dir,
        name_prefix="SAC"
    )
    model.learn(total_timesteps=timesteps, callback=[WandbLoggingCallback(episode_length=50), checkpoint_callback])

    # Save the trained model
    model.save(f"{model_save_dir}/sac_agent")

    # Close the environment
    env.close()

    wandb.finish()

if __name__ == "__main__":
    main()