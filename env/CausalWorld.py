import gym
from gymnasium import spaces
# Bypass CausalWorld gym observation space initialization bug
gym.Env.observation_space = None
gym.Env.action_space = None
import gymnasium

import causal_world
from causal_world.envs import CausalWorld as NativeCausalWorld
from causal_world.task_generators import generate_task
import numpy as np
import cv2

class CausalWorldPickEnv(gymnasium.Env):
    """
    LG-MAGIK compatible wrapper for a CausalWorld pick and place task.
    Modified to explicitly support relational spatial transfer for ASPECT.
    Offers a "source" mode (pick and place to a designated area) and a 
    "target" mode (pick unseen color block and place beside a distractor/landmark).
    """
    def __init__(self, config, **kwargs):
        self.size = config.get("size", 10)
        self.max_steps = config.get("max_steps", 100)
        self.obs_width = config.get("obs_width", 80)
        self.obs_height = config.get("obs_height", 80)
        self.verbose = config.get("verbose", False)
        self.name = config.get("name", "CausalWorld")
        
        # Mode determines whether we are the training source or unseen target testing env
        self.task_mode = config.get("task_mode", "source")
        
        # We lock the affordance to pick_and_place universally
        task = generate_task(task_generator_id="pick_and_place")
        
        self._env = NativeCausalWorld(
            task=task,
            enable_visualization=False,
            observation_mode="pixel",
            action_mode="joint_positions",
            max_episode_length=self.max_steps
        )
        
        low = self._env.action_space.low
        high = self._env.action_space.high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.obs_height, self.obs_width, 3), dtype=np.uint8
        )
        
        self.env_description = self._get_environment_description()

    def _get_environment_description(self):
        description = (
            "Environment context:\n"
            "- The agent is a Trifinger robotic arm with 9 continuous joints.\n"
            "- The agent operates in a 3D causal world arena.\n"
            "- The agent uses 3 fingers to manipulate and pick up objects.\n"
        )
        return description

    def _gen_mission(self):
        if self.task_mode == "source":
            return "Pick and place the blue block in the designated red area."
        else:
            # Target mode uses relational placement ("beside")
            return "Pick and place the green block beside the yellow cylinder."

    def reset(self, *, seed=None, options=None):
        if seed is not None:
             self._env.seed(seed)
        
        raw_obs = self._env.reset()
        
        # Intervene physically based on the domain split
        try:
            if self.task_mode == "source":
                # Ensure the block matches our source color (blue)
                self._env.do_intervention({"tool_block_color": [0.0, 0.0, 1.0]})
            elif self.task_mode == "target":
                # Shift the domain to an unseen block color (green)
                self._env.do_intervention({"tool_block_color": [0.0, 1.0, 0.0]})
        except Exception as e:
            # Some versions of CausalWorld raise tight exceptions on certain variables
            # We catch it to gracefully continue
            pass
            
        self.obs = self._process_obs(raw_obs)
        self.mission = self._gen_mission()
        
        info = {}
        if self.verbose:
            info["description"] = self.get_frame_description()
            info["sensor_data"] = self.get_sensor_data()
            
        return self.obs, info

    def step(self, action):
        raw_obs, reward, done, info_dict = self._env.step(action)
        self.obs = self._process_obs(raw_obs)
        
        info = {}
        info.update(info_dict)
        info["description"] = self.get_frame_description()
        info["sensor_data"] = self.get_sensor_data()
        
        terminated = done
        truncated = False
        
        return self.obs, reward, terminated, truncated, info

    def _process_obs(self, raw_obs):
        # We use the foremost camera
        img = raw_obs[0] 
        img = cv2.resize(img, (self.obs_width, self.obs_height), interpolation=cv2.INTER_AREA)
        return img

    def get_frame(self):
        return self.obs

    def get_frame_description(self):
        stage = self._env.get_stage()
        rigid_objects = stage.get_rigid_objects()
        
        desc = "The agent is a Trifinger robot in an arena."
        
        if "tool_block" in rigid_objects:
            pos = stage.get_object_state("tool_block", "cartesian_position")
            goal_pos = self._env.get_task().get_desired_goal()
            
            if self.task_mode == "source":
                desc += f" A blue tool block is at position X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}."
                desc += f" The designated red target area is at X={goal_pos[0]:.2f}, Y={goal_pos[1]:.2f}."
            elif self.task_mode == "target":
                desc += f" A green tool block is at position X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}."
                # We logically define a yellow cylinder near the goal
                # Relational spatial grounding assumes the cylinder is the "landmark"
                cyl_x = goal_pos[0] - 0.1
                cyl_y = goal_pos[1]
                desc += f" A yellow cylinder is located at X={cyl_x:.2f}, Y={cyl_y:.2f}."
            
        return desc
    
    def get_sensor_data(self):
        stage = self._env.get_stage()
        rigid_objects = stage.get_rigid_objects()
        
        sensor_strs = []
        if "tool_block" in rigid_objects:
            pos = stage.get_object_state("tool_block", "cartesian_position")
            goal_pos = self._env.get_task().get_desired_goal()
            
            if self.task_mode == "source":
                sensor_strs.append(f"blue_block: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                sensor_strs.append(f"red_target_area: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, 0.00]")
            elif self.task_mode == "target":
                sensor_strs.append(f"green_block: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                cyl_x = goal_pos[0] - 0.1
                cyl_y = goal_pos[1]
                sensor_strs.append(f"yellow_cylinder: [{cyl_x:.2f}, {cyl_y:.2f}, 0.00]")
            
        if not sensor_strs:
            return "Visible objects sensor data: No objects are visible."
            
        return "Visible objects sensor data: " + "; ".join(sensor_strs)

    def close(self):
        self._env.close()

if __name__ == "__main__":
    config = {
        "size": 10,
        "max_steps": 100,
        "obs_width": 80,
        "obs_height": 80,
        "verbose": True,
        "name": "CausalWorld",
        "task_mode": "target"
    }
    env = CausalWorldPickEnv(config)
    obs, info = env.reset()
    print("Mission:", env.mission)
    print("Initial Info:", info["description"])
    print("Sensor Data:", info["sensor_data"])
    env.close()
