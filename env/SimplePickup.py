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
from collections import Counter
from omegaconf import DictConfig
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj
from minigrid.core.world_object import  Ball, Wall
from minigrid.wrappers import RGBImgPartialObsWrapper
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
        return "Fetch the Red and Green balls and avoid others"

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
    
        self.mission = self._gen_mission()
        
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
        agent_position = (view_size // 2, view_size - 1)

        descriptions, distances = [], []
        saw_wall = False

        for i in range(view_size):
            for j in range(view_size):
                obj_type, obj_color, obj_state = image[i, j]
                if obj_type == OBJECT_TO_IDX['empty']:
                    continue
                desc = self.get_object_name(obj_type, obj_color, obj_state)

                if "wall" in desc:
                    saw_wall = True
                    continue  # skip walls in detailed listing

                # Manhattan distance
                distance = abs(i - agent_position[0]) + abs(j - agent_position[1])
                descriptions.append(desc)
                distances.append((distance, desc))

        Initial_description = ""
        if self.layout == 0 and saw_wall:
            Initial_description = "Agent is in a room surrounded by blue walls."
        elif self.layout == 1 and saw_wall:
            Initial_description = "Agent is in a room surrounded by grey walls."
        
        # Case 1: only walls in view
        if not descriptions and saw_wall:
            return Initial_description

        # Case 2: no walls, no objects (just floor)
        if not descriptions and not saw_wall:
            return Initial_description + " Agent sees nothing."

        # Case 3: objects visible
        obj_counts = Counter(descriptions)
        seen_list = []
        for desc, count in obj_counts.items():
            parts = desc.split()
            if count == 1:
                seen_list.append(f"a {desc}")
            else:
                if len(parts) == 2:  # color + object
                    color, obj = parts
                    plural = 'boxes' if obj == 'box' else 'keys' if obj == 'key' else obj + 's'
                    seen_list.append(f"{count} {color} {plural}")
                else:
                    seen_list.append(f"{count} {desc}s")

        # Make phrase more natural
        if len(seen_list) == 1:
            object_phrase = seen_list[0]
        else:
            object_phrase = ", ".join(seen_list[:-1]) + " and " + seen_list[-1]

        # --- Closest object(s) logic ---
        distances.sort(key=lambda x: x[0])  # sort by distance
        min_dist = distances[0][0]
        closest_objs = [desc for d, desc in distances if d == min_dist]

        if len(seen_list) > 1:
            final_desc = Initial_description + f" Agent sees {object_phrase}."
        else:
            final_desc = Initial_description + f" Agent sees only {object_phrase}."

        unique_objs = list(dict.fromkeys(closest_objs))
        
        if len(unique_objs) > 1:
            final_desc += f" Agent is equidistant to both {', '.join(unique_objs[:-1])} and {unique_objs[-1]}."
        elif len(seen_list) > 1:
            final_desc += f" Agent is nearest to {unique_objs[-1]}"

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

        
@hydra.main(version_base=None, config_path="../configs/env", config_name="SimplePickup")
def main(args: DictConfig) -> None:
    # type_list = [(i, j) for i in range(0, 6) for j in range(0, 2)]
    collect_data = True
    env = SimplePickup(args, mode="collect_data")
    if collect_data:
        env = RGBImgPartialObsWrapper(env, tile_size=args.tile_size)
        
    paired_data = []

    def rollout(env, remaining_steps, delay=0.0, collect_data=False):
        obs, info = env.reset()
        paired_data.append({"frame" : obs["image"], "description" : info["description"]})
        if not collect_data:
            env.render()

        steps = 0
        while steps < remaining_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if collect_data:
                paired_data.append({"frame" : obs["image"], "description" : info["description"]})
            else:
                env.render()

            time.sleep(delay)
            steps += 1

            if terminated or truncated:
                break

        return steps

    # Total number of timesteps to collect
    total_training_data = 100000
    validation_data = 100
    total_collected = 0
    episode = 0

    while total_collected < total_training_data + validation_data:
        print(f"Collecting from episode {episode} (steps so far: {total_collected})")
        steps = rollout(env, remaining_steps=total_training_data + validation_data - total_collected, collect_data=True)
        total_collected += steps
        episode += 1

    env.close()
    
    if collect_data:
        training_data = paired_data[:total_training_data]
        val_data = paired_data[total_training_data:]

        save_dir = f"data/{env.unwrapped.name}/training"
        os.makedirs(save_dir, exist_ok=True)
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        metadata_path = os.path.join(save_dir, "metadata.jsonl")

        with open(metadata_path, "w") as f:
            for i, data in enumerate(training_data):
                # Save original image
                img_orig = Image.fromarray(data["frame"])
                orig_path = os.path.join(images_dir, f"original_{i:06d}.png")
                img_orig.save(orig_path)

                # Write metadata with _file_name keys for image paths
                f.write(json.dumps({
                    "input_image_file_name": f"images/original_{i:06d}.png",
                    "text": data["description"]
                }) + "\n")

        print(f"Saved {len(training_data)} paired training examples to {save_dir}")

        # Save the paired validation dataset in required format
        save_dir = f"data/{env.unwrapped.name}/validation"
        os.makedirs(save_dir, exist_ok=True)
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        metadata_path = os.path.join(save_dir, "metadata.jsonl")

        with open(metadata_path, "w") as f:
            for i, data in enumerate(val_data):
                # Save original image
                img_orig = Image.fromarray(data["frame"])
                orig_path = os.path.join(images_dir, f"original_{i:06d}.png")
                img_orig.save(orig_path)

                # Write metadata with _file_name keys for image paths
                f.write(json.dumps({
                    "input_image_file_name": f"images/original_{i:06d}.png",
                    "text": data["description"]
                }) + "\n")

        print(f"Saved {len(val_data)} paired validation examples to {save_dir}")
        
if __name__ == "__main__":
    main()