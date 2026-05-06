"""
Quick standalone rollout of the stock panda-gym PandaPickAndPlace-v3.
No wrappers, no Hydra — just raw gym.make + random actions.
"""

import numpy as np
import gymnasium as gym
from PIL import Image
import panda_gym  # registers the envs

env = gym.make("PandaStackDense-v3", render_mode="rgb_array", control_type="ee")

print("=== Environment Info ===")
print("Obs space:", env.observation_space)
for k, v in env.observation_space.spaces.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, low={v.low.min():.1f}, high={v.high.max():.1f}")
print("Act space:", env.action_space)
print(f"  shape={env.action_space.shape}, low={env.action_space.low}, high={env.action_space.high}")
print()

NUM_EPISODES = 3
MAX_STEPS = 50

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    print(f"--- Episode {ep + 1} ---")
    print("Reset obs keys:", list(obs.keys()))
    for k, v in obs.items():
        print(f"  {k}: {v}")
    print("Reset info:", info)

    total_reward = 0.0
    for step in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step == 0:
            print(f"  Step 1 obs:")
            for k, v in obs.items():
                print(f"    {k}: {v}")
            print(f"  Step 1 reward: {reward}, term: {terminated}, trunc: {truncated}")
            print(f"  Step 1 info: {info}")

        if terminated or truncated:
            print(f"  Done at step {step + 1} (terminated={terminated}, truncated={truncated})")
            break

    print(f"  Total reward: {total_reward:.4f}")

    # Grab a frame to confirm rendering works
    frame = env.render()
    if frame is not None:
        print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")
    print()

env.close()
print("=== Done ===")
