import os
import math
import pyglet
pyglet.options['headless'] = True
pyglet.options['headless_device'] = 0
import hydra
import pickle
import random
import itertools
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from gymnasium import spaces
from typing import Optional, Tuple
from gymnasium.core import ObsType
from gymnasium import utils

from miniworld.entity import Ball, Box, MeshEnt, COLOR_NAMES
from miniworld.envs.roomobjects import RoomObjects
from miniworld.miniworld import MiniWorldEnv
import sys
sys.path.append(".")
from architectures.common_utils import collect_data, save_dataset_for_diffusers

class MedKit(MeshEnt):
    """
    A custom entity representing a medkit in the MiniWorld environment.
    """
    def __init__(self, height, **kwargs):
        self.color = 'red'
        super().__init__(mesh_name="medkit", height=height, static=False, **kwargs)
        
class Duckie(MeshEnt):
    """
    A custom entity representing a Duckie in the MiniWorld environment.
    """
    def __init__(self, height, **kwargs):
        self.color = 'yellow'
        super().__init__(mesh_name="duckie", height=height, static=False, **kwargs)
        
class Key(MeshEnt):
    """
    Key the agent can pick up, carry, and use to open doors
    """

    def __init__(self, height, color="yellow"):
        self.color = color
        assert color in COLOR_NAMES
        super().__init__(mesh_name=f"key_{color}", height=height, static=False)
        
class Ball(MeshEnt):
    """
    Ball (sphere) the agent can pick up and carry
    """

    def __init__(self, color, size=0.6):
        self.color = color
        assert color in COLOR_NAMES
        super().__init__(mesh_name=f"ball_{color}", height=size, static=False)
        
OBJECT_TO_ENITTY = {"box" : Box(color="blue", size=0.6),
                    "ball" : Ball(color="green", size=0.8),
                    "duckie" : Duckie(height=0.8),
                    "medkit" : MedKit(height=0.8)}

COLOR_TO_OBJECT = {"blue" : "box",
                   "green" : "ball",
                   "yellow" : "duckie",
                   "red" : "medkit"}

OBJECT_TO_COLOR = {"box" : "blue",
                   "ball" : "green",
                   "duckie" : "yellow",
                   "medkit" : "red"}

INDEX_TO_COLOR = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'red'}

class PickObjectEnv(MiniWorldEnv):
    """
    A custom MiniWorld environment with two layout type : asphalt and grass.
    - Dense reward based on distance to the red box; episode terminates when near any box.
    """
    
    def __init__(self, config, **kwargs):
        """
        Initialize the environment.
        
        Args:
            allowed_types (tuple): Tuple indicating the combination of the layout and the objects
            size (int): Size of the room (default: 10).
            **kwargs: Additional arguments passed to the parent class.
        """
        size = config["size"]
        max_steps=config.get("max_steps")
        if max_steps is None:
            self.max_steps = 4 * size * size
        else:
            self.max_steps = max_steps
        self.objects = list(config.objects)
        self.reward_objects = list(config.reward_objects)
        self.rewarding_object_colors = [OBJECT_TO_ENITTY[obj].color for obj in self.reward_objects]
        self.layout = config.layout
        self.size = config["size"]
        kwargs["obs_width"] = config["obs_width"]
        kwargs["obs_height"] = config["obs_height"]
        kwargs["window_width"] = config["obs_width"]*10
        kwargs["window_height"] = config["obs_height"]*10
        self.verbose = config.get("verbose", False)
        super().__init__(max_episode_steps=self.max_steps, **kwargs)
        self.action_space = spaces.Discrete(self.actions.pickup + 1)
        self.env_description = self._get_environment_description()
        self.env_name = self.set_env_name()
        self.mission = self._gen_mission(self.objects, self.reward_objects, self.layout)
        self.name = config.get("name", "MiniWorld")
        
        # --- Add state for reward shaping and define reward constants ---
        self.dist_to_target = None
        self.REWARD_PICK_SUCCESS = 10.0
        self.REWARD_PICK_FAIL = -10.0
        self.REWARD_TIME_PENALTY = 0.00
        self.REWARD_SHAPING_SCALE = 1.0
        
    def _get_environment_description(self):
        """
        Returns a textual description of the environment dynamics, object affordances,
        agent capabilities, and scene variability.
        This text will be appended to the LLM prompt for imagination reasoning.
        """
        description = (
            "Environment context:\n"
            "- The agent operates in a partially observable 3D gridworld-like room. The agent sees a portion of the room.\n"
            "- At the start of each episode, the agent and objects are randomly initialised in the environment.\n"
            "- The agent can perform the following actions: rotate left/right, move forward/backward, and pick up objects.\n"
            "- The environment may contains different objects of different color.\n"
            "- The agent task is to pick/avoid the objects according to the mission string.\n"
            "- Since, the observation is partial, the agent can explore the environment by moving around to find the objects to pick.\n"
            "- Once one object is picked, the object dissapears from the scene and it is added to agent's inventory, which it can hold forever. This doesn't prevent picking another object later.\n"
            "- The agent can store multiple objects in it's inventory."
            "- Non-interactive elements (walls, floor, background) cannot be acted upon.\n"
            "- The agent receives a reward upon successfully completing the Target task "
            "(for example, picking the specified object from specified room).\n"
            "- Each episode ends once the Target task is completed or a maximum step limit is reached."
        )
        return description
    
    def set_env_name(self):
        for obj in self.reward_objects: assert obj in self.objects, f"Reward object {obj} must be in {self.objects}"
        rewarding_objects = ""
        non_rewarding_object = ""
        reward_objects = self.reward_objects
        objects = self.objects
        if len(reward_objects) == 1:
            rewarding_objects = reward_objects[0].replace(" ", "").capitalize()
            non_rewarding_object = list(set(objects)-set(reward_objects))[0]
            non_rewarding_object = non_rewarding_object.replace(" ", "").capitalize()
        else:
            rewarding_objects = f"{reward_objects[0].replace(' ', '').capitalize()}{reward_objects[1].replace(' ', '').capitalize()}"
        room_color = self.layout.capitalize()
        if not non_rewarding_object:
           return f"Pick{rewarding_objects}Room{room_color}"
        else:
            return f"Pick{rewarding_objects}Avoid{non_rewarding_object}Room{room_color}"

    def _gen_mission(self, objects:list[str], reward_objects:list[str], layout:str) -> str:
        rewarding_objects = ""
        non_rewarding_object = ""
        for rewarding_object in reward_objects: assert rewarding_object in objects, f"{rewarding_object} must be in {objects}"
        if len(reward_objects) == 1:
            rewarding_objects = reward_objects[0]
            non_rewarding_object = list(set(objects)-set(reward_objects))[0]
        else:
            rewarding_objects = f"{reward_objects[0]} and {reward_objects[1]}"
        floor, wall = layout.split("/")
        if not non_rewarding_object:
           return f"Pick {OBJECT_TO_COLOR[rewarding_objects[0]]} {rewarding_objects[0]} and {OBJECT_TO_COLOR[rewarding_objects[1]]} {rewarding_objects[1]} from {floor} floor and {wall} wall"
        else:
            return f"Pick the {OBJECT_TO_COLOR[rewarding_objects]} {rewarding_objects} and avoid {OBJECT_TO_COLOR[non_rewarding_object]} {non_rewarding_object} from {floor} floor and {wall} wall"

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        self.obs, info = super().reset(seed=seed, options=options)
        if self.verbose:
            info["description"] = self.get_frame_description(self.obs)
        self.agent.cam_pitch = -30 * np.pi / 180 #This place the camera -30 degrees downwards and agent can see nearest objects
        self.agent.cam_height = 0.75
        self.reward_object_colors = self.rewarding_object_colors.copy()
        
        # --- EDIT: Reset the distance tracker ---
        self.dist_to_target = None
        
        return self.obs, info
    
    def _add_layout(self):
        floor, wall = self.layout.split("/")
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex=wall,
            floor_tex=floor,
            no_ceiling=True,
        )
        
    def _gen_world(self):
        """
        Generate the world with a room, floor type, and objects.
        """
        self._add_layout()
        for obj in self.objects:
            if obj in OBJECT_TO_ENITTY:
                self.place_entity(OBJECT_TO_ENITTY[obj])
            else:
                raise ValueError(f"Object {obj} not recognized. Available objects: {list(OBJECT_TO_ENITTY.keys())}")
        self.place_agent()
        # Calculate the center of the room
        center_x = self.size / 2
        center_z = self.size / 2 

        # Place the agent in the middle of the room
        self.agent.pos[0] = center_x
        self.agent.pos[2] = center_z
        
        # Use self.np_random for generating random numbers within the environment
        self.agent.dir = self.np_random.uniform(-np.pi, np.pi)

    def step(self, action):
        """
        Execute one step in the environment.
        """
        obs, _, terminated, truncated, info = super().step(action)
        self.obs = obs

        # Start with a small time penalty to encourage efficiency
        reward = self.REWARD_TIME_PENALTY

        # --- Use get_class() to determine visible objects ---
        visible_objects_vec = self.get_class(self.obs)

        # Create a set of colors that are currently visible for efficient lookup
        visible_colors = {
            INDEX_TO_COLOR[i] for i, is_visible in enumerate(visible_objects_vec[:-1]) if is_visible == 1
        }

        # Find the closest visible *rewarding* object
        min_dist = float('inf')
        closest_target_visible = False

        # Only iterate through entities if there are visible colors
        if visible_colors:
            for ent in self.entities:
                # Check if the entity is a rewarding object AND its color is in the visible set
                if hasattr(ent, 'color') and ent.color in self.reward_object_colors and ent.color in visible_colors:
                    delta = ent.pos - self.agent.pos
                    dist = np.sqrt(delta[0]**2 + delta[2]**2)
                    if dist < min_dist:
                        min_dist = dist
                    closest_target_visible = True

        # If a target is visible, provide a distance-based shaping reward
        if closest_target_visible:
            if self.dist_to_target is not None:
                # Reward is proportional to how much closer the agent got
                reward_shaping = self.REWARD_SHAPING_SCALE * (self.dist_to_target - min_dist)
                reward += reward_shaping
            # Update the distance for the next step
            self.dist_to_target = min_dist
        else:
            # If no target is visible, reset the distance tracker
            self.dist_to_target = None

        # Check for pickup action and apply large terminal rewards
        if self.agent.carrying:
            if self.agent.carrying.color in self.reward_object_colors:
                # Overwrite previous rewards with a large success reward
                reward = self.REWARD_PICK_SUCCESS
                self.reward_object_colors.remove(self.agent.carrying.color)
            else:
                # Overwrite previous rewards with a large failure penalty
                reward = self.REWARD_PICK_FAIL

            # Clean up the picked object
            self.entities.remove(self.agent.carrying)
            obs = self.render_obs()
            self.agent.carrying = None

        # Check for termination condition (all rewarding objects collected)
        if not self.reward_object_colors:
            terminated = True

        description = self.get_frame_description(obs)
        if self.verbose:
            info["description"] = description
        return obs, reward, terminated, truncated, info
    
    def get_class(self, obs=None, thresholds=None):
        # This function remains unchanged
        if obs is None:
            obs = self.obs
        if thresholds is None:
            thresholds = {
                'red': {'lower': np.array([120, 0, 0]), 'upper': np.array([255, 60, 60])},
                'green': {'lower': np.array([0, 120, 0]), 'upper': np.array([20, 255, 20])},
                'blue': {'lower': np.array([0, 0, 120]), 'upper': np.array([20, 20, 255])},
                'yellow': {'lower': np.array([120, 120, 0]), 'upper': np.array([255, 255, 20])},
            }
        detected = []
        for color, bounds in thresholds.items():
            mask = np.all((obs >= bounds['lower']) & (obs <= bounds['upper']), axis=-1)
            if np.sum(mask) > 2:
                detected.append(color)
                
        obj_cls = [0] * (len(COLOR_TO_OBJECT) + 1)  # +1 for 'no object' class
        for i, color in enumerate(COLOR_TO_OBJECT.keys()):
            if color in detected:
                obj_cls[i] = 1

        if sum(obj_cls)==0:
            obj_cls[-1] = 1
        return np.array(obj_cls)

    def get_frame_description(self, obs=None):
        # This function remains unchanged
        if obs is None:
            obs = self.obs

        obj_cls = self.get_class(obs)
        color_to_index = {'blue': 0, 'green': 1, 'yellow': 2, 'red': 3}
        detected_colors = [color for color, idx in color_to_index.items() if obj_cls[idx] == 1]

        floor_tex, wall_tex = self.layout.split("/")

        description = f"The agent is in a room with {floor_tex} floor and {wall_tex} walls."

        if len(detected_colors) == 0:
            description += " No objects are visible in the current view."
            return description

        for color in detected_colors:
            ent = None
            for e in self.entities:
                if hasattr(e, 'color') and e.color == color and e is not self.agent:
                    ent = e
                    break
            if ent is None:
                continue 

            delta = ent.pos - self.agent.pos
            dist = np.sqrt(delta[0]**2 + delta[2]**2)

            bearing_ent = np.arctan2(delta[2], delta[0])
            agent_bearing = np.arctan2(self.agent.dir_vec[2], self.agent.dir_vec[0])
            rel_angle_rad = bearing_ent - agent_bearing
            rel_angle_rad = (rel_angle_rad + np.pi) % (2 * np.pi) - np.pi
            angle_deg = np.degrees(rel_angle_rad)

            if abs(angle_deg) < self.agent.cam_fov_y//6:
                dir_str = "in front"
            elif self.agent.cam_fov_y//6 <= abs(angle_deg) < self.agent.cam_fov_y//3:
                if angle_deg > 0:
                    dir_str = "slightly to the right"
                else:
                    dir_str = "slightly to the left"
            elif self.agent.cam_fov_y//3 <= abs(angle_deg) <= self.agent.cam_fov_y//2 + 1:
                if angle_deg > 0:
                    dir_str = "to the right"
                else:
                    dir_str = "to the left"
            else:
                if angle_deg > 0:
                    dir_str = "to the far right"
                else:
                    dir_str = "to the far left"
            object_name = COLOR_TO_OBJECT[color]
            description += f" A {color} {object_name} is found {dir_str} at angle {abs(angle_deg):.3g} at a distance of {dist:.1f} units."

        return description
    
    def get_frame(self):
        top_view = self.render_top_view()
        agent_view = self.render_obs()
        return np.concatenate((top_view, agent_view), axis=1)
    
@hydra.main(version_base=None, config_path="../config/env", config_name="MiniWorld.yaml")
def main(args: DictConfig) -> None:
    # type_list = [(i, j) for i in range(0, 6) for j in range(0, 2)]
    is_collect_data = True
    args.verbose = True
    env = PickObjectEnv(args)
    # Total number of timesteps to collect
    total_training_data = 160000
    validation_data = 100
    paired_data, episode = collect_data(env, total_training_data + validation_data)
    env.close()
    
    # --- Dataset Saving Logic ---
    if is_collect_data:
        training_data = paired_data[:total_training_data]
        val_data = paired_data[total_training_data:]

        # --- Save the training dataset in the required image/text pair format ---
        save_dir_train = f"data/{env.name}/training_images"
        os.makedirs(save_dir_train, exist_ok=True)
        
        # Save the training dataset
        print("\n--- Saving Training Dataset ---")
        save_dataset_for_diffusers(training_data, save_dir_train)

        save_dir_val = f"data/{env.name}/validation_images"
        os.makedirs(save_dir_val, exist_ok=True)
        
        # Save the validation dataset
        print("\n--- Saving Validation Dataset ---")
        save_dataset_for_diffusers(val_data, save_dir_val)

        
if __name__ == "__main__":
    main()