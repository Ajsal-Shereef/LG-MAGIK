import os
import json
import time
import hydra
import pickle
import random
import itertools
import numpy as np

from PIL import Image
from gymnasium import spaces
from omegaconf import DictConfig
from minigrid.core.grid import Grid
from collections import Counter, defaultdict
import sys
sys.path.append(".")
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Wall
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from architectures.common_utils import collect_data, save_dataset_for_images
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle
from minigrid.core.constants import OBJECT_TO_IDX, COLORS, COLOR_TO_IDX, STATE_TO_IDX

class Key(WorldObj):
    def __init__(self, color: str = "blue"):
        super().__init__("key", color)

    def can_pickup(self):
        return True
    
    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("ball", color)

    def can_pickup(self):
        return True
    
    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Box(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains

    def can_pickup(self):
        return True
    
    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(pos[0], pos[1], self.contains)
        return True


NAME_TO_OBJECT = {"ball" : Ball, "key" : Key, "box" : Box}

class FakeWall(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("fakewall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class SimplePickup(MiniGridEnv):
    def __init__(self, config):
        size=config["size"]
        max_steps=config["max_steps"]
        render_mode=config["render_mode"]
        # Register fakewall if not already registered
        if "fakewall" not in OBJECT_TO_IDX:
            OBJECT_TO_IDX["fakewall"] = max(OBJECT_TO_IDX.values()) + 1
        self.objects = list(config.objects)
        self.reward_objects = list(config.reward_objects)
        self.wall_colors = list(config.wall_color)
        if max_steps is None:
            self.max_steps = 4 * size * size
        else:
            self.max_steps = max_steps
            
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.objects, self.reward_objects, self.wall_colors],
        )
        
        super().__init__(
            grid_size=size,
            max_steps=self.max_steps,
            see_through_walls=True,
            agent_view_size=config.agent_view_size,
            render_mode=render_mode,
            mission_space=mission_space,
            highlight=config.highlight
        )
        self.action_space = spaces.Discrete(4)
        self.env_description = self._get_environment_description()
        self.env_name = self.set_env_name()
        
    def _get_environment_description(self):
        """
        Returns a textual description of the environment dynamics, object affordances,
        agent capabilities, and scene variability.
        This text will be appended to the LLM prompt for imagination reasoning.
        """
        description = (
            "Environment context:\n"
            "- The agent operates in a partially observable 2D gridworld-like room. The agent sees a portion of the room.\n"
            "- At the start of each episode, the agent and objects are randomly initialised in the environment.\n"
            "- The agent can perform the following actions: rotate left, rotate right, move forward, and pick up objects that are in front.\n"
            "- The environment may contains different objects such are balls, keys, boxes of different color.\n"
            "- The agent task is to pick/avoid the objects according to the mission string.\n"
            "- Since, the observation is partial, the agent can explore the environment by moving around to find the objects to pick.\n"
            "- Once one object is picked, the object dissapears from the scene and it is added to agent's inventory, which it can hold forever. This doesn't prevent picking another object later.\n"
            "- The agent can store multiple objects in it's inventory, up to a maximum of two objects"
            "- Non-interactive elements (walls, floor, background) cannot be acted upon.\n"
            "- The agent receives a reward upon successfully completing the Target task "
            "(for example, picking the specified object).\n"
            "- Each episode ends once the Target task is completed or a maximum step limit is reached."
        )
        return description
    
    def set_env_name(self):
        reward_objects = self.reward_objects[0]
        objects = self.objects[0]
        if len(reward_objects) == 1:
            rewarding_objects = reward_objects[0].replace(" ", "").capitalize()
            Non_rewarding_object = list(set(objects)-set(reward_objects))[0]
            non_rewarding_object = Non_rewarding_object.replace(" ", "").capitalize()
        else:
            rewarding_objects = f"{reward_objects[0].replace(' ', '').capitalize()}{reward_objects[1].replace(' ', '').capitalize()}"
        room_color = self.wall_colors[0].capitalize()
        # ---- Add a variable to save and later evaluate the performance of the agent ----
        self.agent_performance = {"rewarding_objects" : dict.fromkeys(self.reward_objects[0], 0),
                                  "non_rewarding_objects" : dict.fromkeys([Non_rewarding_object.lower()], 0)}
        if not non_rewarding_object:
           return f"Pick{rewarding_objects}Room{room_color}"
        else:
            return f"Pick{rewarding_objects}Avoid{non_rewarding_object}Room{room_color}"

    @staticmethod
    def _gen_mission(objects:list[str], reward_objects:list[str], wall_colors:list[str]) -> str:
        rewarding_objects = ""
        non_rewarding_object = ""
        for rewarding_object in reward_objects: assert rewarding_object in objects, f"{rewarding_object} must be in {objects}"
        if len(reward_objects) == 1:
            rewarding_objects = reward_objects[0]
            non_rewarding_object = list(set(objects)-set(reward_objects))[0]
        else:
            rewarding_objects = f"{reward_objects[0]} and {reward_objects[1]}"
        room_color = wall_colors
        if not non_rewarding_object:
           return f"Pick {rewarding_objects} from {room_color} room"
        else:
            return f"Pick the {rewarding_objects} and avoid {non_rewarding_object} from {room_color} room"

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["description"] = self.get_description(obs)
        return obs, info

    def _add_layout_walls(self, layout_id, w, h):
        """Place walls based on the layout ID."""
        if layout_id == 'blue':
            # Layout-specific wall placement
            self.grid.horz_wall(0, 0, w, obj_type=FakeWall)
            self.grid.horz_wall(0, 0 + h - 1, w, obj_type=FakeWall)
            self.grid.vert_wall(0, 0, h, obj_type=FakeWall)
            self.grid.vert_wall(0 + w - 1, 0, h, obj_type=FakeWall)
            # Layout 1: vertical wall in the middle with a door
            self.wall = 'blue_wall'
            # for y in range(1, self.height - 1):
            #     if y != self.height // 2:
            #         self.grid.set(self.width // 2, y, FakeWall())
        elif layout_id == 'grey':
            # Layout-specific wall placement
            self.grid.horz_wall(0, 0, w, obj_type=Wall)
            self.grid.horz_wall(0, 0 + h - 1, w, obj_type=Wall)
            self.grid.vert_wall(0, 0, h, obj_type=Wall)
            self.grid.vert_wall(0 + w - 1, 0, h, obj_type=Wall)
            self.wall = 'grey_wall'
            # Layout 2: vertical fake wall in the middle with a door
            # for y in range(1, self.height - 1):
            #     if y != self.height // 2:
            #         self.grid.set(self.width // 2, y, Wall())

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # self.grid.wall_rect(0, 0, width, height)
        self.wall_color = random.choice(self.wall_colors) 
        self.index_id = random.randint(0, len(self.objects)-1)

        for color in self.objects[self.index_id]:assert color in self.objects[self.index_id], f"Reward color {color} must be in {self.objects[self.index_id]}"
        if len(self.reward_objects[self.index_id]) == 1:
            rewarding_color = self.reward_objects[self.index_id][0]
            non_rewarding_colors = list(set(self.objects[self.index_id]) - set(self.reward_objects[self.index_id]))[0]
            self.mission = f"Pick the {rewarding_color} ball and avoid {non_rewarding_colors} ball from {self.wall_color} room"
        else:
            self.mission = f"Pick both {self.reward_objects[self.index_id][0]} and {self.reward_objects[self.index_id][1]} ball from {self.wall_color} room"
            
        self._add_layout_walls(self.wall_color, width, height)
        # Place agent
        self.place_agent()
    
        self.rewarding_objects_class = []
        self.rewarding_objects_color = []
        for obj in self.reward_objects[self.index_id]:
            color, object_name = obj.split(" ")
            object_class = NAME_TO_OBJECT.get(object_name.lower())
            self.rewarding_objects_class.append(type(object_class(color)))
            self.rewarding_objects_color.append(color)
        
        # Place objects at random positions
        for object in self.objects[self.index_id]:
            color, object_name = object.split(" ")
            object_class = NAME_TO_OBJECT.get(object_name.lower())
            self.place_obj(
                object_class(color),
                reject_fn=lambda _, pos: self.grid.get(*pos) is not None
            )
        

    def get_object_name(self, obj_type, obj_color, obj_state):
        """Returns a string description of an object based on its type, color, and state."""
        #Invert the dictionary
        IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
        IDX_TO_OBJECTS = {v: k for k, v in OBJECT_TO_IDX.items()}
        IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
        # Map indices to names
        color_name = IDX_TO_COLOR.get(int(obj_color), "unknown color)")
        object_name = IDX_TO_OBJECTS.get(int(obj_type), "unknown object")
        state_name = IDX_TO_STATE.get(int(obj_state), "")
        # Construct a descriptive name
        return f"{color_name} {object_name}".strip()
        
    def get_description(self, observation):
        """Generate short, instruction-like natural captions for the observation."""
        image = observation['image']
        view_size = image.shape[0]
        agent_x = view_size // 2
        agent_y = view_size - 1
        agent_position = (agent_x, agent_y)

        descriptions = []
        saw_wall = False

        directions_map = {
            "top_left": [],
            "bottom_left": [],
            "front": [],
            "top_right": [],
            "bottom_right": []
        }

        for x in range(view_size):  # left to right (formerly i)
            for y in range(view_size):  # far to near (formerly j)
                obj_type, obj_color, obj_state = image[x, y]
                if obj_type == OBJECT_TO_IDX['empty']:
                    continue
                desc = self.get_object_name(obj_type, obj_color, obj_state)

                if "wall" in desc:
                    saw_wall = True
                    continue  # skip walls in detailed listing

                # Determine direction
                dx = x - agent_x
                if dx == 0:
                    direction = "front"
                else:
                    mid_y = view_size // 2
                    is_top = y < mid_y
                    prefix = "top" if is_top else "bottom"
                    suffix = "left" if dx < 0 else "right"
                    direction = prefix + "_" + suffix

                # Manhattan distance
                dist = abs(dx) + abs(y - agent_y)
                if dist == 0:  # skip agent itself if present
                    continue

                descriptions.append(desc)
                directions_map[direction].append((dist, desc, x, y))

        initial_description = ""
        if saw_wall:
            initial_description = f"Agent is in a room surrounded by {self.wall_color} walls."

        # Case 1: only walls in view
        if not descriptions and saw_wall:
            return initial_description + " No other objects can be seen."

        # Case 2: no walls, no objects (just floor)
        if not descriptions and not saw_wall:
            return initial_description + " Agent sees nothing."

        # Build directional phrases
        dir_phrases = []
        for dir_key, prefix in [
            ("top_left", "to the top left"),
            ("top_right", "to the top right"),
            ("front", "in front"),
            ("bottom_left", "to the bottom left"),
            ("bottom_right", "to the bottom right")
        ]:
            obj_list = directions_map[dir_key]
            if not obj_list:
                continue
            # Sort by distance
            obj_list.sort(key=lambda t: t[0])

            sub_phrases = []
            for dist, desc, x, y in obj_list:
                unit_str = "unit" if dist == 1 else "units"
                sub_phrases.append(f"a {desc} which is {dist} {unit_str} apart at ({x},{y})")

            if len(sub_phrases) == 1:
                phrase = f"{prefix}, {sub_phrases[0]}"
            else:
                phrase = f"{prefix}, " + ", ".join(sub_phrases[:-1]) + " and " + sub_phrases[-1]

            dir_phrases.append(phrase)

        # Construct main description
        if len(dir_phrases) == 1:
            sees_part = f"Agent sees {dir_phrases[0]}."
        else:
            sees_part = f"Agent sees {', '.join(dir_phrases[:-1])} and {dir_phrases[-1]}."

        final_desc = initial_description + (" " if initial_description else "") + sees_part

        return final_desc

    def get_class(self, observation=None):
        
        if observation is None:
            observation = self.obs
            
        object_caption, _ = self.get_description(observation)

        # Initialize multi-label vector
        object_vector = np.zeros(5, dtype=np.float32)

        # Detect presence of specific colored balls
        colors = ['red', 'blue', 'green', 'yellow']

        for i, color in enumerate(colors):
            if f'{color} ball' in object_caption:
                object_vector[i] = 1.0
        if sum(object_vector)==0:
            object_vector[-1] = 1.0
        return object_vector
        # # Determine environment layout (based on wall)
        # if self.wall == "blue_wall":
        #     layout_vector = np.array([1.0, 0.0], dtype=np.float32)
        # else:
        #     layout_vector = np.array([0.0, 1.0], dtype=np.float32)
        # # Final label vector: [red, blue, green, yellow, layout0, layout1]
        # return np.concatenate([object_vector, layout_vector], axis=-1)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["description"] = self.get_description(obs)
        self.obs = obs['image']
        # Custom reward logic for picking up objects
        if action == self.actions.pickup and self.carrying is not None:
            if type(self.carrying) in self.rewarding_objects_class:
                color = self.carrying.color
                if color in self.rewarding_objects_color[self.index_id]:
                    reward = 1.0
                    self.agent_performance["rewarding_objects"][f"{color} {self.carrying.type}"] += 1
                else:
                    reward = -1.0
                    self.agent_performance["non_rewarding_objects"][f"{color} {self.carrying.type}"] += 1
            else:
                reward = -1.0
            # Drop the object after reward is given
            self.carrying = None
            terminated = True  # Optionally end episode after pickup

        return obs, reward, terminated, truncated, info
    
    def get_performance_metric(self):
        return self.agent_performance
    

@hydra.main(version_base=None, config_path="../config/env", config_name="SimplePickup")
def main(args: DictConfig) -> None:
    # type_list = [(i, j) for i in range(0, 6) for j in range(0, 2)]
    is_collect_data = True
    env = SimplePickup(args)
    if is_collect_data:
        env = RGBImgPartialObsWrapper(env, tile_size=args.tile_size)
        env = ImgObsWrapper(env) 
    paired_data = []
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
        save_dir_train = f"data/{env.unwrapped.name}/training_images"
        os.makedirs(save_dir_train, exist_ok=True)
        
        # Save the training dataset
        print("\n--- Saving Training Dataset ---")
        save_dataset_for_images(training_data, save_dir_train)

        save_dir_val = f"data/{env.unwrapped.name}/validation_images"
        os.makedirs(save_dir_val, exist_ok=True)
        
        # Save the validation dataset
        print("\n--- Saving Validation Dataset ---")
        save_dataset_for_images(val_data, save_dir_val)

        
if __name__ == "__main__":
    main()