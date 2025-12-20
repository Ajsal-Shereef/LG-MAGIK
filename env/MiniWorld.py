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

from miniworld.entity import Entity, Ball, Box, MeshEnt, COLOR_NAMES
from miniworld.math import X_VEC, Y_VEC, Z_VEC, gen_rot_matrix
from miniworld.envs.roomobjects import RoomObjects
from miniworld.miniworld import MiniWorldEnv
# Map of color names to RGB values
from pyglet.gl import (
    GL_LINES,
    GL_QUADS,
    GL_TEXTURE_2D,
    GL_TRIANGLES,
    glBegin,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glNormal3f,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glScalef,
    glTexCoord2f,
    glTranslatef,
    glVertex3f,
)
import sys
sys.path.append(".")
from architectures.common_utils import collect_data, save_dataset_for_images

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
                    "duckie" : Duckie(height=0.6),
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

class Agent(Entity):
    def __init__(self):
        super().__init__()

        # Distance between the camera and the floor
        self.cam_height = 0.75

        # Camera up/down angles in degrees
        # Positive angles tilt the camera upwards
        self.cam_pitch = -30 * np.pi / 180

        # Vertical field of view in degrees
        self.cam_fov_y = 60

        # Bounding cylinder size for the agent
        self.radius = 0.4
        self.height = 1.6

        # Object currently being carried by the agent
        self.carrying = None

    @property
    def cam_pos(self):
        """
        Camera position in 3D space
        """

        rot_y = gen_rot_matrix(Y_VEC, self.dir)
        cam_disp = np.array([self.cam_fwd_disp, self.cam_height, 0])
        cam_disp = np.dot(cam_disp, rot_y)

        return self.pos + cam_disp

    @property
    def cam_dir(self):
        """
        Camera direction (lookat) vector

        Note: this is useful even if just for slight domain
        randomization of camera angle
        """

        rot_z = gen_rot_matrix(Z_VEC, self.cam_pitch * math.pi / 180)
        rot_y = gen_rot_matrix(Y_VEC, self.dir)

        dir = np.dot(X_VEC, rot_z)
        dir = np.dot(dir, rot_y)

        return dir

    def randomize(self, params, rng):
        params.sample_many(
            rng,
            self,
            [
                "cam_height",
                "cam_fwd_disp",
                "cam_pitch",
                "cam_fov_y",
            ],
        )
        # self.radius = params.sample(rng, 'bot_radius')

    def render(self):
        """
        Draw the agent
        """

        # Note: this is currently only used in the top view
        # Eventually, we will want a proper 3D model

        p = self.pos + Y_VEC * self.height
        dv = self.dir_vec * self.radius
        rv = self.right_vec * self.radius

        p0 = p + dv
        p1 = p + 0.75 * (rv - dv)
        p2 = p + 0.75 * (-rv - dv)

        glColor3f(1, 0, 0)
        glBegin(GL_TRIANGLES)
        glVertex3f(*p0)
        glVertex3f(*p2)
        glVertex3f(*p1)
        glEnd()

        """
        glBegin(GL_LINE_STRIP)
        for i in range(20):
            a = (2 * math.pi * i) / 20
            pc = p + dv * math.cos(a) + rv * math.sin(a)
            glVertex3f(*pc)
        glEnd()
        """

    def step(self, delta_time):
        pass

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
        self.size = config["size"]
        max_steps=config.get("max_steps")
        kwargs["obs_width"] = config["obs_width"]
        kwargs["obs_height"] = config["obs_height"]
        kwargs["window_width"] = config["obs_width"]*10
        kwargs["window_height"] = config["obs_height"]*10
        self.verbose = config.get("verbose", False)
        if max_steps is None:
            self.max_steps = 4 * self.size * self.size
        else:
            self.max_steps = max_steps
        self.objects = list(config.objects)
        self.reward_objects = list(config.reward_objects)
        self.layout = config.layout
        
        super().__init__(max_episode_steps=self.max_steps, **kwargs)
        self.action_space = spaces.Discrete(self.actions.pickup + 1)
        self.env_description = self._get_environment_description()
        self.env_name = self.set_env_name()
        self.reset_metrices()
        self.mission = self._gen_mission()
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
        self.current_reward_objects = [item for sublist in self.reward_objects for item in sublist]
        objects = [item for sublist in self.objects for item in sublist]
        layout = self.layout
        for obj in self.current_reward_objects: assert obj in objects, f"Reward object {obj} must be in {objects}"
        self.current_non_rewarding_obj_list = list(set(objects)-set(self.current_reward_objects))
        rewarding_objects = ""
        non_rewarding_object = ""
        for obj in self.current_reward_objects:
            rewarding_objects += obj.capitalize()
        for obj in self.current_non_rewarding_obj_list:
            non_rewarding_object += obj.capitalize()

        room_color = ""
        for room in layout:
            room_color += room.capitalize()

        if not non_rewarding_object:
           return f"Pick{rewarding_objects}Room{room_color}"
        else:
            return f"Pick{rewarding_objects}Avoid{non_rewarding_object}Room{room_color}"
        
    def reset_metrices(self):
        # ---- Add a variable to save and later evaluate the performance of the agent ----
        self.agent_performance = {"rewarding_objects" : dict.fromkeys(self.current_reward_objects, 0),
                                  "non_rewarding_objects" : dict.fromkeys(self.current_non_rewarding_obj_list, 0)}

    def _gen_mission(self) -> str:
        # Flatten lists
        reward_objects_flat = [item for sublist in self.reward_objects for item in sublist]
        objects_flat = [item for sublist in self.objects for item in sublist]
        
        # --- LAYOUT LOGIC UPDATE ---
        # We iterate through self.layout (the list) to handle 1 or multiple rooms
        location_descriptions = []
        for layout_option in self.layout:
            floor, wall = layout_option.split("/")
            location_descriptions.append(f"{floor} floor and {wall} wall")
        
        # Join descriptions with " or " if there are multiple, otherwise it's just the one.
        # Example: "grass floor and concrete wall or asphalt floor and brick_wall wall"
        location_str = " or ".join(location_descriptions)
        
        # --- OBJECT LOGIC ---
        non_rewarding_obj_list = list(set(objects_flat) - set(reward_objects_flat))
        
        for rewarding_object in reward_objects_flat: 
            assert rewarding_object in objects_flat, f"{rewarding_object} must be in {objects_flat}"
            
        rewarding_object_description = []
        for obj in reward_objects_flat:
            rewarding_object_description.append(f"{OBJECT_TO_COLOR[obj]} {obj}")
            
        rewarding_object_str = " and ".join(rewarding_object_description)
        
        non_rewarding_object_description = []
        for obj in non_rewarding_obj_list:
            non_rewarding_object_description.append(f"{OBJECT_TO_COLOR[obj]} {obj}")
            
        non_rewarding_object_str = " and ".join(non_rewarding_object_description)
        
        if not non_rewarding_obj_list:
            # Scenario: Pick two reward objects
            return f"Pick {rewarding_object_str} from the room with {location_str}"
        else:
            # Scenario: Pick one reward, avoid distractor
            return f"Pick the {rewarding_object_str} and avoid {non_rewarding_object_str} from the room with {location_str}"

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.np_random if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(
            rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"]
        )

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max("forward_step")

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)
            
        self.agent.cam_height = 0.75

        # Camera up/down angles in degrees
        # Positive angles tilt the camera upwards
        self.agent.cam_pitch = -30 * np.pi / 180

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        self.obs = self.render_obs()
        info = {}
        if self.verbose:
            info["description"] = self.get_frame_description(self.obs)
        # --- Reset the distance tracker ---
        self.dist_to_target = None
        
        return self.obs, info
    
    def _add_layout(self):
        layout = random.choice(self.layout)
        floor, wall = layout.split("/")
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
        index = random.randint(0,len(self.objects)-1)
        objects = self.objects[index]
        for obj in objects:
            if obj in OBJECT_TO_ENITTY:
                self.place_entity(OBJECT_TO_ENITTY[obj])
            else:
                raise ValueError(f"Object {obj} not recognized. Available objects: {list(OBJECT_TO_ENITTY.keys())}")
        self.place_agent()
        
        self.reward_object_colors = [OBJECT_TO_COLOR[obj] for obj in self.reward_objects[index]]
        
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
                self.agent_performance["rewarding_objects"][f"{COLOR_TO_OBJECT[self.agent.carrying.color]}"] += 1
                self.reward_object_colors.remove(self.agent.carrying.color)
            else:
                # Overwrite previous rewards with a large failure penalty
                reward = self.REWARD_PICK_FAIL
                self.agent_performance["non_rewarding_objects"][f"{COLOR_TO_OBJECT[self.agent.carrying.color]}"] += 1

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
    
    def get_performance_metric(self):
        return self.agent_performance
    
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
        
        layout = random.choice(self.layout)
        floor_tex, wall_tex = layout.split("/")

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
    
    # def get_frame_description(self, obs=None):
    #     # This function remains unchanged
    #     if obs is None:
    #         obs = self.obs

    #     obj_cls = self.get_class(obs)
    #     color_to_index = {'blue': 0, 'green': 1, 'yellow': 2, 'red': 3}
    #     detected_colors = [color for color, idx in color_to_index.items() if obj_cls[idx] == 1]

    #     floor_tex, wall_tex = self.layout.split("/")

    #     description = f"The agent is in a room with {floor_tex} floor and {wall_tex} walls."

    #     if len(detected_colors) == 0:
    #         description += " No objects are visible in the current view."
    #         return description

    #     for color in detected_colors:
    #         ent = None
    #         for e in self.entities:
    #             if hasattr(e, 'color') and e.color == color and e is not self.agent:
    #                 ent = e
    #                 break
    #         if ent is None:
    #             continue 

    #         delta = ent.pos - self.agent.pos
    #         dist = np.sqrt(delta[0]**2 + delta[2]**2)

    #         bearing_ent = np.arctan2(delta[2], delta[0])
    #         agent_bearing = np.arctan2(self.agent.dir_vec[2], self.agent.dir_vec[0])
    #         rel_angle_rad = bearing_ent - agent_bearing
    #         rel_angle_rad = (rel_angle_rad + np.pi) % (2 * np.pi) - np.pi
    #         angle_deg = np.degrees(rel_angle_rad)

    #         if abs(angle_deg) < self.agent.cam_fov_y//6:
    #             dir_str = "in front"
    #         elif self.agent.cam_fov_y//6 <= abs(angle_deg) < self.agent.cam_fov_y//3:
    #             if angle_deg > 0:
    #                 dir_str = "slightly to the right"
    #             else:
    #                 dir_str = "slightly to the left"
    #         elif self.agent.cam_fov_y//3 <= abs(angle_deg) <= self.agent.cam_fov_y//2 + 1:
    #             if angle_deg > 0:
    #                 dir_str = "to the right"
    #             else:
    #                 dir_str = "to the left"
    #         else:
    #             if angle_deg > 0:
    #                 dir_str = "to the far right"
    #             else:
    #                 dir_str = "to the far left"
            
    #         # --- Add distance description ---
    #         if dist < 1.0:
    #             dist_str = "very close"
    #         elif dist < 2.5:
    #             dist_str = "nearby"
    #         elif dist < 4:
    #             dist_str = "at a medium distance"
    #         else:
    #             dist_str = "far away"
    #         # --- End of new block ---

    #         object_name = COLOR_TO_OBJECT[color]
            
    #         # --- Updated description string ---
    #         description += f" A {color} {object_name} is found {dist_str}, {dir_str}."

    #     return description
    
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
    total_training_data = 150000
    validation_data = 0
    paired_data, episode = collect_data(env, total_training_data + validation_data)
    env.close()
    
    # --- Dataset Saving Logic ---
    if is_collect_data:
        training_data = paired_data[:total_training_data]
        # val_data = paired_data[total_training_data:]

        # --- Save the training dataset in the required image/text pair format ---
        save_dir_train = f"data/{env.name}/Random/vae/domain_2"
        os.makedirs(save_dir_train, exist_ok=True)
        
        # Save the training dataset
        print("\n--- Saving Training Dataset ---")
        save_dataset_for_images(training_data, save_dir_train)

        # save_dir_val = f"data/{env.name}/Random/test"
        # os.makedirs(save_dir_val, exist_ok=True)
        
        # # Save the validation dataset
        # print("\n--- Saving Validation Dataset ---")
        # save_dataset_for_images(val_data, save_dir_val)

        
if __name__ == "__main__":
    main()