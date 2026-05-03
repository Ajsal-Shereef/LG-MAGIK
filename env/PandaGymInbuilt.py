import numpy as np
import gymnasium as gym
from gymnasium import spaces
from omegaconf import DictConfig

import pybullet
import pybullet_utils.bullet_client as _bc
import inspect

_orig_sig = inspect.signature(_bc.BulletClient.__init__)
if "options" not in _orig_sig.parameters:
    _OrigBulletClient = _bc.BulletClient

    class _PatchedBulletClient(_OrigBulletClient):
        """BulletClient that silently accepts (and ignores) `options`."""
        def __init__(self, connection_mode=None, options="", **kwargs):
            if connection_mode is None:
                connection_mode = pybullet.DIRECT
            self._client = pybullet.connect(connection_mode)
            self._shapes = {}

        def __getattr__(self, name):
            attribute = getattr(pybullet, name)
            if callable(attribute):
                import functools
                @functools.wraps(attribute)
                def wrapper(*args, **kw):
                    return attribute(*args, physicsClientId=self._client, **kw)
                return wrapper
            return attribute

        def __del__(self):
            if hasattr(self, '_client') and self._client >= 0:
                try:
                    pybullet.disconnect(self._client)
                except Exception:
                    pass

    _bc.BulletClient = _PatchedBulletClient

import panda_gym

class PandaGymInbuiltEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        
        self.reward_type = config.get("reward_type", "dense")
        self.use_her = config.get("use_her", False)
        
        if self.reward_type == "dense":
            gym_id = "PandaPickAndPlaceDense-v3"
        else:
            gym_id = "PandaPickAndPlace-v3"
            
        self._inner = gym.make(gym_id, render_mode="rgb_array")
        
        self.action_space = self._inner.action_space
        
        # Original panda-gym obs space is Dict(observation:19, achieved_goal:3, desired_goal:3)
        if self.use_her:
            self.observation_space = self._inner.observation_space
        else:
            flat_dim = 19 + 3 + 3
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(flat_dim,), dtype=np.float32)
            
        self.env_name = "PandaGymInbuilt"
        self.mission = "Pick up the block and place it on the target."
        self.reset_metrices()

    def reset_metrices(self):
        self.agent_performance = {
            "successful_place": 0,
            "episodes": 0,
        }

    def get_performance_metric(self):
        return self.agent_performance
        
    def _flatten_obs(self, obs_dict):
        return np.concatenate([
            obs_dict["observation"], 
            obs_dict["achieved_goal"], 
            obs_dict["desired_goal"]
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self._inner.reset(seed=seed, options=options)
        
        self.agent_performance["episodes"] += 1
        
        info_out = {
            "description": self.get_description(obs_dict),
            "sensor_data": self.get_sensor_data(obs_dict),
            "is_success": info.get("is_success", False),
        }
        
        return obs_dict if self.use_her else self._flatten_obs(obs_dict), info_out

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self._inner.step(action)
        
        if info.get("is_success", False):
            self.agent_performance["successful_place"] += 1
            
        info_out = {
            "description": self.get_description(obs_dict),
            "sensor_data": self.get_sensor_data(obs_dict),
            "is_success": info.get("is_success", False),
        }
        
        return obs_dict if self.use_her else self._flatten_obs(obs_dict), reward, terminated, truncated, info_out

    def get_description(self, obs_dict=None):
        return "Inbuilt PandaGym PickAndPlace environment."

    def get_sensor_data(self, obs_dict=None):
        return "Sensor data: []"

    def get_frame(self):
        frame = self._inner.render()
        if frame is None:
            return np.zeros((84, 84, 3), dtype=np.uint8)
        return frame.astype(np.uint8)
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._inner.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    def close(self):
        try:
            self._inner.close()
        except Exception:
            pass

    @property
    def unwrapped(self):
        return self
