import numpy as np
from stable_baselines3 import SAC
from argparse import ArgumentParser
from configparser import ConfigParser
from helper_functions.utils import load_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'reacher', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--test_episode", type=int, default = 10, help = "Number of testing episodes")
args = parser.parse_args()
parser = ConfigParser()


# Load trained model
model = SAC.load("model_weights/reacher/2025-02-21_18-21-06_S1R8SI/sac_agent.zip")

# Create the Reacher environment
if args.env_name == "reacher":
    from env.reacher import Reacher4TargetsEnv
    env_config = load_config("config/reacher.yml")
    env = Reacher4TargetsEnv(goal_target_idx=env_config["goal_target_idx"], max_steps = env_config["max_steps"], render_mode="rgb_array")
else:
    raise NotImplementedError

video_folder = f"videos/{args.env_name}"
video_length = env_config["max_steps"]

# Loop to record multiple episodes
for episode in range(args.test_episode):
    env = DummyVecEnv([lambda: env])  # Wrap for SB3 compatibility
    # Define a unique video filename for each episode
    video_filename = f"{video_folder}/{episode+1}"
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=video_filename)

    # Reset environment and start recording
    obs = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions
        obs, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            obs = env.reset()

    # Close the environment to finalize the video
    env.close()
