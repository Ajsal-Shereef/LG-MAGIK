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
from architectures.common_utils import rollout, collect_data
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj
from minigrid.core.world_object import  Ball, Wall
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from minigrid.utils.rendering import fill_coords, point_in_rect
from minigrid.core.constants import OBJECT_TO_IDX, COLORS, COLOR_TO_IDX, STATE_TO_IDX


class FakeWall(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("fakewall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class SimplePickup(MiniGridEnv):
    def __init__(self, config, mode="train"):
        size=config["size"]
        max_steps=config["max_steps"]
        if mode == "train":
            self.allowed_types=config["agent_training_allowed_types"]
        elif mode == "collect_data":
            self.allowed_types=config["data_collection_allowed_types"]
        else:
            self.allowed_types=config["test_allowed_types"]
        render_mode=config["render_mode"]
        # Register fakewall if not already registered
        if "fakewall" not in OBJECT_TO_IDX:
            OBJECT_TO_IDX["fakewall"] = max(OBJECT_TO_IDX.values()) + 1
        self.all_colors = ["red", "blue", "green", "yellow"]
        self.ball_combinations = list(itertools.combinations(range(4), 2))  # 6 combinations
        self.current_type = (1, 1)
        
        if max_steps is None:
            self.max_steps = 4 * size * size
        else:
            self.max_steps = max_steps
            
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
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
        self.name = 'SimplePickup'

    @staticmethod
    def _gen_mission():
        return "Fetch the Red ball"

    def reset(self, *, seed=None, options=None):
        self.current_type = random.choice(self.allowed_types)
        self.layout = self.current_type[-1]
        obs, info = super().reset(seed=seed, options=options)
        info["description"] = self.get_description(obs)
        return obs, info

    def _add_layout_walls(self, layout_id, w, h):
        """Place walls based on the layout ID."""
        if layout_id == 0:
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
        elif layout_id == 1:
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
    
        ball_index, layout_id = self.current_type
        ball_ids = self.ball_combinations[ball_index]
        ball_colors = [self.all_colors[i] for i in ball_ids]
        
        if 'red' in ball_colors:
            self.mission = "Fetch the Red ball"
        if 'green' in ball_colors:
            self.mission = "Fetch the Green ball"
        # Define gate position based on layout
        gate_pos = None
        # gate_pos = (width // 2, height // 2)  # vertical wall with gate
        # Layout-specific wall placement
        self._add_layout_walls(layout_id, width, height)
        # Place agent
        self.place_agent()
    
        # Place balls at random positions excluding the gate position
        for color in ball_colors:
            self.place_obj(
                Ball(color),
                reject_fn=lambda _, pos: self.grid.get(*pos) is not None or pos == gate_pos
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
        if self.layout == 0 and saw_wall:
            initial_description = "Agent is in a room surrounded by blue walls."
        elif self.layout == 1 and saw_wall:
            initial_description = "Agent is in a room surrounded by grey walls."

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
        # Custom reward logic for picking up balls
        if action == self.actions.pickup and self.carrying is not None:
            if isinstance(self.carrying, Ball):
                color = self.carrying.color
                if color in ["red", "green"]:
                    reward = 1.0
                else:
                    reward = -1.0
                # Drop the ball after reward is given
                self.carrying = None
                terminated = True  # Optionally end episode after pickup

        return obs, reward, terminated, truncated, info
    
def save_dataset_for_diffusers(dataset, save_dir):
    """
    Saves a dataset of images and captions in the format expected by diffusers.

    This creates a directory with an 'images' subfolder and a 'metadata.jsonl' file.

    Args:
        dataset (list): A list of dictionaries, where each dict has "frame" and "description".
        save_dir (str): The path to the root directory where the dataset will be saved.
    """
    if not dataset:
        print("Warning: No data to save.")
        return
        
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    metadata_entries = []
    
    print(f"Saving {len(dataset)} items to {save_dir}...")
    for i, item in enumerate(dataset):
        try:
            # Create a Pillow Image from the numpy array
            img = Image.fromarray(item["frame"])
            
            # Define image filename and save it
            base_filename = f"{i:06d}.png"
            img_path = os.path.join(images_dir, base_filename)
            img.save(img_path)
            
            # Create metadata entry
            # The file_name must be relative to the root of the dataset directory
            metadata_entry = {
                "file_name": os.path.join("images", base_filename),
                "text": item["description"]
            }
            metadata_entries.append(metadata_entry)

        except Exception as e:
            print(f"Error processing item {i}: {e}")

    # Write the metadata.jsonl file
    metadata_path = os.path.join(save_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Successfully saved dataset with {len(metadata_entries)} entries.")
    print(f"Images saved in: {images_dir}")
    print(f"Metadata saved in: {metadata_path}")

        
@hydra.main(version_base=None, config_path="../config/env", config_name="SimplePickup")
def main(args: DictConfig) -> None:
    # type_list = [(i, j) for i in range(0, 6) for j in range(0, 2)]
    is_collect_data = True
    env = SimplePickup(args, mode="collect_data")
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
        save_dataset_for_diffusers(training_data, save_dir_train)

        save_dir_val = f"data/{env.unwrapped.name}/validation_images"
        os.makedirs(save_dir_val, exist_ok=True)
        
        # Save the validation dataset
        print("\n--- Saving Validation Dataset ---")
        save_dataset_for_diffusers(val_data, save_dir_val)

        
if __name__ == "__main__":
    main()