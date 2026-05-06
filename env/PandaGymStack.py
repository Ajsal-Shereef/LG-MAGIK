"""
PandaGymInbuiltEnv: Thin passthrough wrapper around panda-gym's built-in
PandaPickAndPlace-v3 / PandaPickAndPlaceDense-v3.

All observations, rewards, and dynamics are exactly as panda-gym provides.
This wrapper only adds:
  - Config acceptance (DictConfig from Hydra)
  - .mission / .env_name / .get_frame() for train_agent_zoo.py compatibility
  - Simple performance metrics
"""

import random
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig

import panda_gym  # noqa: registers gym envs


# Pool of random mission strings
_MISSION_POOL = [
    "Pick up the cube and place it at the target location.",
    "Grab the green block and move it to the goal position.",
    "Lift the object and set it down on the target.",
    "Grasp the cube, then transport it to the indicated spot.",
    "Move the block from the table to the highlighted goal.",
    "Use the gripper to pick the cube and deliver it to the target.",
    "Relocate the cube to the marked destination.",
    "Take the block and drop it at the goal area.",
    "Seize the object and carry it over to the target zone.",
    "Grip the green cube and deposit it on the goal marker.",
]


class PandaGymStackEnv(gym.Wrapper):
    """
    Thin wrapper around stock panda-gym PandaPickAndPlace(Dense)-v3.

    Native observations (Dict with observation/achieved_goal/desired_goal)
    are passed through unchanged.
    """

    def __init__(self, config: DictConfig, **kwargs):
        reward_type  = config.get("reward_type", "sparse")
        control_type = config.get("control_type", "ee")
        max_steps    = config.get("max_steps", 200)

        if reward_type == "dense":
            gym_id = "PandaStackDense-v3"
        else:
            gym_id = "PandaStack-v3"

        inner = gym.make(
            gym_id,
            render_mode="rgb_array",
            control_type=control_type,
            max_episode_steps=max_steps,
        )
        super().__init__(inner)

        # --- ASPECT / train_agent_zoo.py interface ---
        self.env_name = "PandaGymStack"
        self.mission  = "Stack lighter object on heavier object"
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Minimal overrides to inject info keys expected by callbacks
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_count += 1
        info.setdefault("description", self._describe(obs))
        info.setdefault("sensor_data", self._sensor(obs))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.setdefault("description", self._describe(obs))
        info.setdefault("sensor_data", self._sensor(obs))
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Required by VideoRolloutCallback
    # ------------------------------------------------------------------
    def get_frame(self):
        frame = self.env.render()
        if frame is None:
            return np.zeros((480, 480, 3), dtype=np.uint8)
        return frame.astype(np.uint8)

    # ------------------------------------------------------------------
    # Required by HER (delegates to inner task)
    # ------------------------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.unwrapped.task.compute_reward(
            achieved_goal, desired_goal, info
        )

    # ------------------------------------------------------------------
    # Simple text descriptions for DataCollectorCallback
    # ------------------------------------------------------------------
    def _describe(self, obs):
        ee   = obs["observation"][:3]
        cube = obs["achieved_goal"]
        goal = obs["desired_goal"]
        return (
            f"Gripper at ({ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}). "
            f"Cube at ({cube[0]:.2f}, {cube[1]:.2f}, {cube[2]:.2f}). "
            f"Goal at ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})."
        )

    def _sensor(self, obs):
        ee   = obs["observation"][:3]
        cube = obs["achieved_goal"]
        goal = obs["desired_goal"]
        return (
            f"ee: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]; "
            f"cube: [{cube[0]:.3f}, {cube[1]:.3f}, {cube[2]:.3f}]; "
            f"goal: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]"
        )

    @property
    def unwrapped(self):
        return self
