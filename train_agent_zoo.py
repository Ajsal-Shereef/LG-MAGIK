import sys
sys.path.append('.')
import pyglet
pyglet.options['headless'] = True
pyglet.options['headless_device'] = 0
import os
import wandb
import pickle
import hydra
import numpy as np
import warnings
import gymnasium as gym
from PIL import Image
from collections import deque
from omegaconf import DictConfig, OmegaConf
from architectures.common_utils import create_dump_directory, save_gif
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


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
    def __init__(self, verbose=1):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.current_episode_reward = 0
        self.episode_count = 0

    def _on_step(self):
        # This callback is called *after* the env.step()
        # self.locals["rewards"] is an array of (n_envs,)
        reward = self.locals.get("rewards", [0])[0] # Get reward for the first env
        done = self.locals.get("dones", [False])[0]

        self.current_episode_reward += reward

        if done:
            self.episode_count += 1
            log_data = {
                "env_step": self.num_timesteps,
                "episode_reward": self.current_episode_reward,
                "episode": self.episode_count
            }
            wandb.log(log_data, step=self.num_timesteps)
            self.current_episode_reward = 0
            
        return True


class VideoRolloutCallback(BaseCallback):
    """
    Record rollout videos during training.
    """
    def __init__(self, dump_dir: str, rollout_freq: int, episode_length: int, fps: int = 10, keep_last_n: int = 5):
        super(VideoRolloutCallback, self).__init__()
        self.dump_dir = dump_dir
        self.rollout_freq = rollout_freq
        self.episode_length = episode_length
        self.fps = fps
        self.keep_last_n = keep_last_n
        self.saved_rollouts = []

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.rollout_freq == 0:
            print(f"\n[INFO] Performing video rollout at step {self.num_timesteps}...")

            eval_env = self.training_env.envs[0] if hasattr(self.training_env, "envs") else self.training_env
            obs, info = eval_env.reset()
            for episode in range(5):
                frame_array = []
                frame_array.append(eval_env.unwrapped.get_frame())
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    frame_array.append(eval_env.unwrapped.get_frame())
                    done = terminated or truncated
                    if done:
                        obs, info = eval_env.reset()
                        break

                save_name = f"rollout_step_{self.num_timesteps}"
                save_gif(frame_array, episode, f'{self.dump_dir}/{self.num_timesteps}',
                         fps=self.fps, save_name=save_name)
            print("Roll out saved in ", self.dump_dir)
            
            self.saved_rollouts.append(f'{self.dump_dir}/{self.num_timesteps}')
            while len(self.saved_rollouts) > self.keep_last_n:
                old_rollout = self.saved_rollouts.pop(0)
                import shutil
                if os.path.exists(old_rollout):
                    shutil.rmtree(old_rollout)
        return True

# --- DATA COLLECTION CALLBACK ---
class DataCollectorCallback(BaseCallback):
    """
    A custom callback to collect and save interaction data
    (s, info) for all algorithms.
    """
    def __init__(self, save_path: str, saving_func, number_data_to_collect, observation_mode, use_her=False, verbose=0):
        super(DataCollectorCallback, self).__init__(verbose)
        self.save_path = save_path
        # We'll store all the data in these lists
        self.all_states = []
        self.all_description = []
        self.all_sensor_data = []
        self.saving_func = saving_func
        self.number_data_to_collect = number_data_to_collect
        self.observation_mode = observation_mode
        self.use_her = use_her
        
    def _on_step(self) -> bool:
        """
        This method is called at every step for all algorithms.
        """
        # self.model._last_obs is the state (s_t) *before* the action was taken
        # self.locals["infos"] is the info dict (info_t+1)
        
        # Get data from the environment step
        # Get data from the environment step
        s = self.locals["new_obs"]
        infos = self.locals["infos"] # Get info dictionary
        
        # Handle terminal observations
        # If an episode is done, 'new_obs' is the reset observation of the NEW episode.
        # We want the terminal observation of the COMPLETED episode, which is in info['terminal_observation'].
        if self.use_her and isinstance(s, dict):
            keys = ["observation", "achieved_goal", "desired_goal"]
            keys = [k for k in keys if k in s] or sorted(s.keys())
            
            flat_s = np.concatenate([s[k].reshape(s[k].shape[0], -1) for k in keys], axis=-1)
            current_obs = flat_s.copy()
            
            for i, info in enumerate(infos):
                if "terminal_observation" in info:
                    term_obs = info["terminal_observation"]
                    flat_term_obs = np.concatenate([term_obs[k].reshape(-1) for k in keys], axis=-1)
                    current_obs[i] = flat_term_obs
        else:
            current_obs = s.copy()
            for i, info in enumerate(infos):
                if "terminal_observation" in info:
                    current_obs[i] = info["terminal_observation"]

        description = [info.get("description", "") for info in infos]
        sensor_data = [info.get("sensor_data", "") for info in infos]
        
        # Append data. 
        self.all_states.append(np.squeeze(current_obs))
        self.all_description.append(description) # Append infos
        self.all_sensor_data.append(sensor_data)
        
        return True

    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        We'll concatenate and save the data.
        """
        if not self.all_states:
            if self.verbose > 0:
                print("[DataCollector] No data collected.")
            return
            
        if self.verbose > 0:
            print(f"[DataCollector] Training ended. Saving collected data to {self.save_path}")

        def process_data(data_list, observation_mode=None):
            """
            Helper to process list of (n_envs, *shape) arrays.
            Converts [ (n_envs, *shape), (n_envs, *shape), ... ]
            into one large (total_timesteps, *shape) array.
            """
            # 1. Stack rollouts along a new time axis
            data_np = np.stack(data_list, axis=0) # (total_steps, n_envs, *shape)
            # 2. Swap n_envs and steps axes
            if observation_mode == "image" and data_np.ndim == 5:
                data_np = data_np.transpose(1, 0, 3, 4, 2)
            else:
                data_np = data_np.swapaxes(0, 1) 
            # 3. Reshape to (total_timesteps, *shape)
            # This interleaves the data from different envs (e.g., e1_t1, e2_t1, e1_t2, e2_t2)
            # Handle cases where shape is (n_envs, total_steps) -> (total_timesteps,)
            if data_np.ndim == 2:
                 data_np = data_np.reshape(-1)
            else:
                data_np = data_np.reshape(-1, *data_np.shape[2:])
            return data_np

        try:
            all_states = process_data(self.all_states, self.observation_mode)
            all_descriptions = process_data(self.all_description) # Process infos
            all_sensor_data = process_data(self.all_sensor_data)
            total_samples = len(all_states)
            
            # Determine the number of samples to pick
            num_to_sample = min(self.number_data_to_collect, total_samples)
            
            if self.verbose > 0:
                print(f"[DataCollector] Processing {total_samples} total samples.")
                print(f"[DataCollector] Randomly sampling {num_to_sample} data points.")

            # Generate random indices without replacement
            indices = np.random.choice(total_samples, size=num_to_sample, replace=False)
            
            # Select the random samples
            sampled_states = all_states[indices]
            sampled_descriptions = all_descriptions[indices]
            sampled_sensor_data = all_sensor_data[indices]

            # Create the paired list
            paired_data = [
                {'frame': state, 'description': desc, 'sensor_data': sens}
                for state, desc, sens in zip(sampled_states, sampled_descriptions, sampled_sensor_data)
            ]

            # Call the saving function with the paired data and save path
            if self.verbose > 0:
                print(f"[DataCollector] Calling saving function to save {len(paired_data)} pairs...")
            
            self.saving_func(paired_data, self.save_path) # Call the function
            
            if self.verbose > 0:
                print(f"[DataCollector] Successfully saved data.")
                print(f"[DataCollector] Total States shape: {all_states.shape}")
                print(f"[DataCollector] Total Descriptions shape: {all_descriptions.shape}")
                print(f"[DataCollector] Sampled {len(paired_data)} pairs.")
        
        except Exception as e:
            if self.verbose > 0:
                print(f"[DataCollector] Error processing or saving data: {e}")


class CheckpointAndPruneCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", save_replay_buffer: bool = False, save_vecnormalize: bool = False, verbose: int = 0, keep_last_n: int = 5):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.keep_last_n = keep_last_n

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            import glob
            import re
            
            # Prune models
            checkpoint_files = glob.glob(os.path.join(self.save_path, f"{self.name_prefix}_*steps.zip"))
            def extract_step(path):
                match = re.search(r"_(\d+)_steps\.zip", path)
                return int(match.group(1)) if match else 0
            
            checkpoint_files.sort(key=extract_step)
            while len(checkpoint_files) > self.keep_last_n:
                old_file = checkpoint_files.pop(0)
                if os.path.exists(old_file):
                    os.remove(old_file)
                    
            # Prune replay buffers
            if self.save_replay_buffer:
                replay_files = glob.glob(os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_*steps.pkl"))
                def extract_step_rb(path):
                    match = re.search(r"_(\d+)_steps\.pkl", path)
                    return int(match.group(1)) if match else 0
                replay_files.sort(key=extract_step_rb)
                while len(replay_files) > self.keep_last_n:
                    old_file = replay_files.pop(0)
                    if os.path.exists(old_file):
                        os.remove(old_file)
        
        return result


@hydra.main(version_base=None, config_path="config", config_name="train_agent_zoo")
def main(args: DictConfig) -> None:
    # --- ENVIRONMENT SETUP ---
    if args.env.name == "PickEnv":
        from env.PickEnv import PickEnv
        env = PickEnv(args.env, mode=args.env.mode, render_mode="rgb_array")
        mission = env.mission
    elif args.env.name == "MiniWorld":
        from env.MiniWorld import PickObjectEnv
        env = PickObjectEnv(args.env)
        mission = env.unwrapped.mission
    elif args.env.name ==  "SimplePickup":
        from env.SimplePickup import SimplePickup
        env = SimplePickup(args.env)
        from minigrid.wrappers import RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env, tile_size=args.env.tile_size)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        mission = env.unwrapped.mission
    elif args.env.name == "PandaGym":
        from env.PandaGym import PandaGymPickPlaceEnv
        env = PandaGymPickPlaceEnv(args.env, render_mode="rgb_array")
        mission = env.unwrapped.mission
    elif args.env.name == "PandaGymInbuilt":
        import panda_gym
        env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
        mission = "random mission string"
    
    # Setting the mission string
    args.env.mission = mission
     
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="LG-MAGIK",
            name=f"{args.agent_name}_{args.env.name}",
            config=OmegaConf.to_container(args, resolve=True)
        )

    model_save_dir = create_dump_directory(f"{args.save_model_dir}/{args.agent_name}")
    print("[INFO] Model save directory: ", model_save_dir)
    
    # SAving the training config
    config_path = os.path.join(model_save_dir, "config.yaml")
    OmegaConf.save(config=args, f=config_path)

    # --- HYPERPARAMS FROM RL_ZOO3 ---
    import yaml
    import rl_zoo3
    from stable_baselines3 import HerReplayBuffer
    
    zoo_path = os.path.dirname(rl_zoo3.__file__)
    algo_lower = args.agent_name.lower()
    hp_path = os.path.join(zoo_path, 'hyperparams', f'{algo_lower}.yml')
    
    zoo_kwargs = {}
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            data = yaml.safe_load(f)
            # Default to PandaPickAndPlace-v1 if PandaGymInbuilt maps to it
            env_key = "PandaPickAndPlace-v1"
            if env_key in data:
                zoo_kwargs = data[env_key]
                print(f"[INFO] Loaded {args.agent_name} hyperparameters for {env_key} from rl_zoo3.")
            else:
                print(f"[WARNING] No hyperparameters found for {env_key} in {hp_path}.")
    else:
        print(f"[WARNING] rl_zoo3 hyperparams file not found: {hp_path}")

    # Process zoo_kwargs
    kwargs = {}
    
    for k, v in zoo_kwargs.items():
        if k in ["n_timesteps", "policy"]:
            continue
        if isinstance(v, str) and v.startswith("dict("):
            # Safe eval for dict
            try:
                v = eval(v, {"dict": dict})
            except Exception as e:
                print(f"[WARNING] Failed to eval {k}: {v}. Error: {e}")
        elif isinstance(v, str) and v.startswith("["):
            try:
                v = eval(v)
            except:
                pass
        
        if k == "replay_buffer_class" and v == "HerReplayBuffer":
            kwargs["replay_buffer_class"] = HerReplayBuffer
            continue

        kwargs[k] = v

    # --- TRAINING ---
    # timesteps = zoo_kwargs.get("n_timesteps", args.env.total_timestep)
    
    if getattr(args.env, "use_her", False):
        policy = "MultiInputPolicy"
        kwargs["replay_buffer_class"] = HerReplayBuffer
    elif "policy" in zoo_kwargs:
        policy = zoo_kwargs["policy"]
    elif args.env.observation_mode == "feature":
        policy = "MlpPolicy"
    else:
        policy = "CnnPolicy"
        
    if args.env.observation_mode == "feature":
        from architectures.common_utils import save_dataset_for_features
        saving_data_function = save_dataset_for_features
    else:
        from architectures.common_utils import save_dataset_for_images
        saving_data_function = save_dataset_for_images
        
    if args.fine_tune and not os.path.exists(args.fine_tune_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.fine_tune_checkpoint}")

    if args.agent_name == "SAC":
        from stable_baselines3 import SAC
        if args.fine_tune:
            model = SAC.load(args.fine_tune_checkpoint, env=env)
            print(f"[INFO] Loaded SAC model from {args.fine_tune_checkpoint} for fine tuning")
        else:
            model = SAC(policy, env, verbose=1, **kwargs)
    elif args.agent_name == "TQC":
        from sb3_contrib import TQC
        if args.fine_tune:
            model = TQC.load(args.fine_tune_checkpoint, env=env)
            print(f"[INFO] Loaded TQC model from {args.fine_tune_checkpoint} for fine tuning")
        else:
            model = TQC(policy, env, verbose=0, **kwargs)
    elif args.agent_name == "PPO":
        from stable_baselines3 import PPO
        if getattr(args, "use_her", False):
            raise ValueError("HER is not supported for on-policy algorithms like PPO.")
        if args.fine_tune:
            model = PPO.load(args.fine_tune_checkpoint, env=env)
            print(f"[INFO] Loadied PPO model from {args.fine_tune_checkpoint} for fine tuning")
        else:
            model = PPO(policy, env, verbose=1, **kwargs)
    elif args.agent_name == "DQN":
        from stable_baselines3 import DQN
        if args.fine_tune:
            model = DQN.load(args.fine_tune_checkpoint, env=env)
            print(f"[INFO] Loaded DQN model from {args.fine_tune_checkpoint} for fine tuning")
        else:
            model = DQN(policy, env, verbose=1, **kwargs)
    elif args.agent_name == "DDPG":
        from stable_baselines3 import DDPG
        if args.fine_tune:
            model = DDPG.load(args.fine_tune_checkpoint, env=env)
            print(f"[INFO] Loaded DDPG model from {args.fine_tune_checkpoint} for fine tuning")
        else:
            model = DDPG(policy, env, verbose=1, **kwargs)
    elif args.agent_name == "TD3":
        from stable_baselines3 import TD3
        if args.fine_tune:
            model = TD3.load(args.fine_tune_checkpoint, env=env)
            print(f"[INFO] Loaded TD3 model from {args.fine_tune_checkpoint} for fine tuning")
        else:
            model = TD3(policy, env, verbose=1, **kwargs)
    else:
        raise ValueError("Algorithm not supported. Supported Algorithms are DQN, PPO, SAC, DDPG, TD3, TQC") 
        
    if args.use_wandb:
        wandb.watch(model.policy, log="all", log_freq=100)

    # --- CALLBACKS ---
    wandb_callback = WandbLoggingCallback()
    checkpoint_callback = CheckpointAndPruneCallback(
        save_freq=10000,
        save_path=model_save_dir,
        name_prefix=args.agent_name,
        keep_last_n=5
    )
    video_callback = VideoRolloutCallback(
        dump_dir=model_save_dir + '/training_videos/',
        rollout_freq=50000,
        episode_length=args.env.max_steps,
        fps=args.env.fps
    )
    
    # --- ADDED DATA COLLECTION CALLBACK ---
    data_save_path = os.path.join(args.data_dave_dir, args.env.name, "agent")
    use_her = getattr(args.env, "use_her", False)
    data_collector_callback = DataCollectorCallback(save_path=data_save_path, saving_func=saving_data_function, 
                                                    number_data_to_collect=int(args.number_data_to_collect),  
                                                    observation_mode=args.env.observation_mode, use_her=use_her, verbose=1)
    call_backs = [data_collector_callback, checkpoint_callback, video_callback]
    # call_backs = []
    all_callbacks = call_backs + [wandb_callback] if args.use_wandb else call_backs
    
    model.learn(total_timesteps=int(args.env.total_timestep),
                callback=all_callbacks)

    # --- SAVE ---
    model.save(f"{model_save_dir}/{args.agent_name}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
