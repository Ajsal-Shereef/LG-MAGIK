import sys
sys.path.append('.')

import os
import wandb
import pickle
import hydra
import numpy as np
import warnings
import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO
from omegaconf import DictConfig, OmegaConf
from architectures.common_utils import create_dump_directory, save_gif
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

class WandbLoggingCallback(BaseCallback):
    def __init__(self, episode_length, verbose=1):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.episode_length = episode_length
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self):
        # SB3 stores the last reward(s) in "rewards" local (vectorized envs -> array)
        reward = 0
        if "rewards" in self.locals:
            r = self.locals["rewards"]
            # handle vectorized and non-vectorized cases
            try:
                reward = float(np.array(r).ravel()[0])
            except Exception:
                reward = float(r)
        self.episode_rewards.append(reward)

        if self.num_timesteps > 0 and (self.num_timesteps % self.episode_length) == 0:
            episode_reward = np.sum(self.episode_rewards)
            self.episode_count += 1
            log_data = {
                "env_step": self.num_timesteps,
                "episode_reward": float(episode_reward),
                "episode": self.episode_count
            }
            wandb.log(log_data, step=self.num_timesteps)
            self.episode_rewards = []

        return True

class VideoRolloutCallback(BaseCallback):
    """
    Callback to record a video of the agent's performance during training.

    Uses Gymnasium-compatible reset/step signatures.

    :param dump_dir: The directory to save the GIF.
    :param rollout_freq: How often to record a video (in training steps).
    :param episode_length: The maximum length of an episode to record.
    :param fps: The frames per second for the saved GIF.
    """
    def __init__(self, dump_dir: str, rollout_freq: int, episode_length: int, fps: int = 10):
        super(VideoRolloutCallback, self).__init__()
        self.dump_dir = dump_dir
        self.rollout_freq = rollout_freq
        self.episode_length = episode_length
        self.fps = fps

    def _on_step(self) -> bool:
        # When it's time to perform a rollout
        if self.n_calls > 0 and (self.n_calls % self.rollout_freq == 0):
            print(f"\n[INFO] Performing video rollout at step {self.num_timesteps}...")
            # Determine a non-vectorized evaluation env to step through
            eval_env = None
            if hasattr(self.training_env, "envs"):
                # If a VecEnv was used, attempt to use the first wrapped environment
                try:
                    eval_env = self.training_env.envs[0]
                except Exception:
                    eval_env = self.training_env
            else:
                eval_env = self.training_env

            # Do a few episodes
            for episode in range(10):
                frame_array = []
                obs, info = eval_env.reset()
                # obs might already be an image; if it's a dict/tuple handle accordingly
                try:
                    state = np.squeeze(obs, axis=0).transpose(1,2,0)
                except Exception:
                    # fallback: assume obs is already HWC
                    state = obs
                frame_array.append(state)
                for _ in range(self.episode_length):
                    # Use the current model for prediction
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = eval_env.step(action)
                    # Gymnasium returns: obs, reward, terminated, truncated, info
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = bool(terminated or truncated)
                    else:
                        # backward-compatibility with older gym
                        obs, reward, done, info = step_result

                    try:
                        state = np.squeeze(obs, axis=0).transpose(1,2,0)
                    except Exception:
                        state = obs
                    frame_array.append(state)

                    if done:
                        break

                # Save the collected frames as a GIF
                save_name = f"rollout_step_{self.num_timesteps}"
                # ensure directory exists
                os.makedirs(f'{self.dump_dir}/{self.num_timesteps}', exist_ok=True)
                save_gif(frame_array, episode, f'{self.dump_dir}/{self.num_timesteps}', fps=self.fps, save_name=save_name)

        return True


@hydra.main(version_base=None, config_path="../config", config_name="train_agent")
def main(args: DictConfig) -> None:
    # Initialize wandb
    wandb.init(project="LG-MAGIK", name=f"{args.agent.name}_{args.env.name}", config=OmegaConf.to_container(args, resolve=True))

    model_save_dir = create_dump_directory(f"model_weights/{args.agent.name}")
    print("[INFO] Model save directory: ", model_save_dir)
    if args.env.name == "PickEnv":
        from env.PickEnv import PickEnv
        # Ensure the environment is created with a render mode that returns images
        env = PickEnv(args.env, render_mode='rgb_array')

    # Train the agent
    timesteps = args.env.total_timestep

    # Use PPO instead of SAC. Adjust hyperparams as needed.
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=2048,       # rollout length
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.0,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        tensorboard_log=None
    )

    # Log model parameters and gradients using wandb.watch()
    # (watching the policy is fine for monitoring gradients/params)
    wandb.watch(model.policy, log="all", log_freq=100)

    # --- SETUP CALLBACKS ---
    wandb_callback = WandbLoggingCallback(episode_length=50)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_save_dir,
        name_prefix="PPO"
    )
    
    video_callback = VideoRolloutCallback(
        dump_dir=model_save_dir + '/training_videos/',
        rollout_freq=50000,
        episode_length=args.env.max_steps, # Should match your environment's episode length
        fps=30
    )

    # Pass all callbacks to the learn method
    model.learn(total_timesteps=timesteps, callback=[wandb_callback, checkpoint_callback, video_callback])

    # Save the final trained model
    model.save(f"{model_save_dir}/ppo_agent")

    # Close the environment
    try:
        env.close()
    except Exception:
        pass

    wandb.finish()

if __name__ == "__main__":
    main()
