import os
import pickle
import random
import numpy as np

from PIL import Image
from collections import Counter
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper


class RandomObjectsEnv(MiniGridEnv):
    """
    A custom MiniGrid environment with random balls, boxes, and keys.
    No specific goal, just exploration. The state description is added to info.
    """

    def __init__(
        self,
        size=8,
        max_steps: int | None = None,
        num_objects=4,
        tile_size=8,
        **kwargs,
    ):
        self.size = size
        self.tile_size = tile_size
        self.num_objects = num_objects

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            tile_size=tile_size,
            **kwargs,
        )
        
        self.name = "MiniGrid"

    @staticmethod
    def _gen_mission():
        return "Explore the grid with keys, balls, and boxes"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Spawn only these
        object_classes = [Ball, Key, Box]

        for _ in range(self.num_objects):
            cls = random.choice(object_classes)
            color = random.choice(COLOR_NAMES)
            obj = cls(color)
            self.place_obj(obj)

        # Place agent
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # Facing right

        self.mission = "Explore the grid with keys, balls, and boxes"

    def get_object_name(self, obj_type, obj_color, obj_state):
        """Returns a string description of an object based on its type, color, and state."""
        IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
        IDX_TO_OBJECTS = {v: k for k, v in OBJECT_TO_IDX.items()}
        IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

        color_name = IDX_TO_COLOR.get(int(obj_color), "unknown color")
        object_name = IDX_TO_OBJECTS.get(int(obj_type), "unknown object")
        state_name = IDX_TO_STATE.get(int(obj_state), "")

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

                distance = np.sqrt((i - agent_position[0])**2 + (j - agent_position[1])**2)
                descriptions.append(desc)
                distances.append((distance, desc))

        # Case 1: only walls in view
        if not descriptions and saw_wall:
            return "Agent is surrounded by walls."

        # Case 2: no walls, no objects (just floor)
        if not descriptions and not saw_wall:
            return "Agent sees nothing."

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

        # Closest object info
        distances.sort()
        closest_desc = distances[0][1] if distances else None

        if closest_desc:
            return f"Agent is near {object_phrase}."
        else:
            return f"Agent sees {object_phrase}."

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["description"] = self.get_description(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["description"] = self.get_description(obs)
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = RandomObjectsEnv(max_steps=20, size=8, tile_size=64)
    env = RGBImgPartialObsWrapper(env, tile_size=16)

    N = 10
    paired_data = []  # list of dicts: {"frame": , "caption": , "changed_caption": , "changed_frame": }

    obs, info = env.reset()
    base_frame = obs["image"]
    base_caption = info["description"]

    for _ in range(N):
        # Save base observation
        frame = obs["image"]
        caption = info["description"]

        # ---- Generate a changed environment ----
        # Copy environment to make a modification
        env2 = RandomObjectsEnv(max_steps=1, size=8, tile_size=64)
        env2 = RGBImgPartialObsWrapper(env2, tile_size=64)
        obs2, info2 = env2.reset()

        # Change rule: pick one object and recolor it
        # For simplicity, just replace the first colored object if exists
        changed_caption = None
        changed_frame = None

        for i in range(env.unwrapped.grid.width):
            for j in range(env.unwrapped.grid.height):
                obj = env.unwrapped.grid.get(i, j)
                if obj is not None and obj.type in ["ball", "box", "key"]:
                    original_color = obj.color
                    # Pick a different color
                    new_color = random.choice([c for c in COLOR_NAMES if c != original_color])
                    obj.color = new_color
                    # Regenerate frame + caption
                    obs2 = env.render(mode="rgb_array")
                    changed_frame = obs2
                    changed_caption = caption.replace(original_color, new_color, 1)
                    break
            if changed_caption:
                break

        if changed_caption is not None and changed_frame is not None:
            paired_data.append({
                "frame": frame,
                "caption": caption,
                "changed_caption": changed_caption,
                "changed_frame": changed_frame,
            })

        # Step environment to get next base observation
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    # Save the paired dataset
    save_dir = f"data/{env.unwrapped.name}_paired"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "paired.pkl"), "wb") as f:
        pickle.dump(paired_data, f)

    print(f"Saved {len(paired_data)} paired examples to {save_dir}")

