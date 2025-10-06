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
from collections import deque
from stable_baselines3 import SAC
from omegaconf import DictConfig, OmegaConf
from architectures.common_utils import create_dump_directory, save_gif
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class FrameStackHW(gym.Wrapper):
    """
    Stack the last k frames along the channel axis.
    Input:  H x W x C
    Output: H x W x (C * k)
    """
    def __init__(self, env, k):
        super(FrameStackHW, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        obs_shape = env.observation_space.shape  # (H, W, C)
        assert len(obs_shape) == 3, "FrameStackHW expects image observations (H,W,C)"
        H, W, C = obs_shape

        # New obs space: (H, W, C*k)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(H, W, C * k),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Concatenate along the last axis (channel axis)
        return np.concatenate(list(self.frames), axis=-1)


class WandbLoggingCallback(BaseCallback):
    def __init__(self, episode_length, verbose=1):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.episode_length = episode_length
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self):
        reward = self.locals.get("rewards", [0])[0]
        self.episode_rewards.append(reward)

        if self.num_timesteps > 0 and self.num_timesteps % self.episode_length == 0:
            episode_reward = np.sum(self.episode_rewards)
            self.episode_count += 1
            log_data = {
                "env_step": self.num_timesteps,
                "episode_reward": episode_reward,
                "episode": self.episode_count
            }
            wandb.log(log_data, step=self.num_timesteps)
            self.episode_rewards = []
        return True


class VideoRolloutCallback(BaseCallback):
    """
    Record rollout videos during training.
    """
    def __init__(self, dump_dir: str, rollout_freq: int, episode_length: int, fps: int = 10):
        super(VideoRolloutCallback, self).__init__()
        self.dump_dir = dump_dir
        self.rollout_freq = rollout_freq
        self.episode_length = episode_length
        self.fps = fps

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.rollout_freq == 0:
            print(f"\n[INFO] Performing video rollout at step {self.num_timesteps}...")

            eval_env = self.training_env.envs[0] if hasattr(self.training_env, "envs") else self.training_env

            for episode in range(5):
                frame_array = []
                obs, info = eval_env.reset()
                frame_array.append(eval_env.render())

                for _ in range(self.episode_length):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    frame_array.append(eval_env.render())

                    if terminated or truncated:
                        break

                save_name = f"rollout_step_{self.num_timesteps}"
                save_gif(frame_array, episode, f'{self.dump_dir}/{self.num_timesteps}',
                         fps=self.fps, save_name=save_name)
        return True


@hydra.main(version_base=None, config_path="../config", config_name="train_agent")
def main(args: DictConfig) -> None:
    # Initialize wandb
    wandb.init(
        project="LG-MAGIK",
        name=f"{args.agent.name}_{args.env.name}",
        config=OmegaConf.to_container(args, resolve=True)
    )

    model_save_dir = create_dump_directory(f"model_weights/{args.agent.name}")
    print("[INFO] Model save directory: ", model_save_dir)

    # --- ENVIRONMENT SETUP ---
    if args.env.name == "PickEnv":
        from env.PickEnv import PickEnv
        def make_env():
            env = PickEnv(args.env, render_mode="rgb_array")
            env = FrameStackHW(env, k=4)
            return env

        env = DummyVecEnv([make_env])   # <-- SB3 works with this

    # --- TRAINING ---
    timesteps = args.env.total_timestep
    if args.env.observation_mode == "feature":
        model = SAC("MlpPolicy", env, verbose=0,
                    buffer_size=int(timesteps / 5), learning_starts=5000)
    else:
        model = SAC("CnnPolicy", env, verbose=0,
                    buffer_size=int(timesteps / 5), learning_starts=5000)

    wandb.watch(model.policy, log="all", log_freq=100)

    # --- CALLBACKS ---
    wandb_callback = WandbLoggingCallback(episode_length=50)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_save_dir,
        name_prefix="SAC"
    )
    video_callback = VideoRolloutCallback(
        dump_dir=model_save_dir + '/training_videos/',
        rollout_freq=50000,
        episode_length=args.env.max_steps,
        fps=30
    )

    model.learn(total_timesteps=timesteps,
                callback=[wandb_callback, checkpoint_callback, video_callback])

    # --- SAVE ---
    model.save(f"{model_save_dir}/sac_agent")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
