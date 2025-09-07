import os
import json
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
        max_num_objects=3,
        tile_size=8,
        highlight=False,
        **kwargs,
    ):
        self.size = size
        self.tile_size = tile_size
        self.max_num_objects = max_num_objects

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            tile_size=tile_size,
            highlight=highlight,
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

        # --- Place the agent at a random empty position ---
        agent_pos = (random.randint(1, width - 2), random.randint(1, height - 2))
        self.agent_pos = agent_pos
        self.agent_dir = random.randint(0, 3)  # random facing direction (0=right,1=down,2=left,3=up)

        # --- Spawn objects ---
        object_classes = [Ball, Key, Box] 
        num_objects = random.randint(1, self.max_num_objects) 
        for _ in range(num_objects): 
            cls = random.choice(object_classes) 
            color = random.choice(COLOR_NAMES) 
            obj = cls(color) 
            self.place_obj(obj)

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

                # Manhattan distance
                distance = abs(i - agent_position[0]) + abs(j - agent_position[1])
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

        # --- Closest object(s) logic ---
        distances.sort(key=lambda x: x[0])  # sort by distance
        min_dist = distances[0][0]
        closest_objs = [desc for d, desc in distances if d == min_dist]

        final_desc = f"Agent sees {object_phrase}."

        if min_dist <= 2:  # only say "near" if really close
            if len(closest_objs) == 1:
                return final_desc + f" Agent is near {closest_objs[0]}."
            else:
                # Deduplicate multiple identical names
                unique_objs = list(dict.fromkeys(closest_objs))
                if len(unique_objs) == 1:
                    return final_desc + f" Agent is near {unique_objs[0]}."
                else:
                    return final_desc + f" Agent is equidistant to {', '.join(unique_objs[:-1])} and {unique_objs[-1]}."

        return final_desc

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["description"] = self.get_description(obs)
        self.obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["description"] = self.get_description(obs)
        self.obs = obs
        return obs, reward, terminated, truncated, info
    
    def get_obs(self):
        return self.unwrapped.get_frame(
            tile_size=self.tile_size, agent_pov=True
        )
        
    
if __name__ == "__main__":
    
    def generate_change_description(env):
        """
        Generate a valid change description based on objects visible in the partial observation.
        Only considers objects in the agent's view.
        """
        # Map indices back to names
        IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
        IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

        image = env.unwrapped.obs['image']  # partial view
        view_size = image.shape[0]
        agent_pos = (view_size // 2, view_size - 1)  # agent position in partial view

        objs = []

        # Collect objects in partial view
        for i in range(view_size):
            for j in range(view_size):
                obj_type, obj_color, obj_state = image[i, j]
                if obj_type in [OBJECT_TO_IDX["ball"], OBJECT_TO_IDX["box"], OBJECT_TO_IDX["key"]]:
                    # distance from agent in partial view (Manhattan)
                    dist = abs(i - agent_pos[0]) + abs(j - agent_pos[1])
                    objs.append(((i, j), obj_type, obj_color, dist))

        if not objs:
            return None  # nothing to change

        # Group by (color, type)
        grouped = {}
        for pos, obj_type, obj_color, dist in objs:
            key = (IDX_TO_COLOR[obj_color], IDX_TO_OBJECT[obj_type])
            grouped.setdefault(key, []).append((pos, dist))

        # Pick a target group
        target_color, target_type = random.choice(list(grouped.keys()))
        candidates = sorted(grouped[(target_color, target_type)], key=lambda x: x[1])

        # Decide distance qualifier
        if len(candidates) == 1:
            distance_phrase = ""  # e.g., "green key"
        elif len(candidates) == 2:
            distance_phrase = random.choice(["closest", "farthest"])
        else:  # 3 or more
            distance_phrase = random.choice(["closest", "middle", "farthest"])

        # Decide new color/type
        change_mode = random.choice(["color", "type", "both"])
        new_color, new_type = target_color, target_type

        if change_mode in ["color", "both"]:
            new_color = random.choice([c for c in COLOR_NAMES if c != target_color])
        if change_mode in ["type", "both"]:
            new_type = random.choice([t for t in ["ball", "box", "key"] if t != target_type])

        # Build description using human-readable names
        if distance_phrase:
            desc = f"change the {distance_phrase} {target_color} {target_type} to {new_color} {new_type}"
        else:
            desc = f"change the {target_color} {target_type} to {new_color} {new_type}"

        return desc

    def apply_change(env, change_desc, caption=None):
        """
        Modify the *partial* observation grid (from gen_obs_grid) according to change_desc
        (e.g. "change the farthest green key to blue ball"), render the edited partial view,
        and return (rgb_image, changed_caption, changed_obs_index).

        - Does NOT modify the full env.unwrapped.grid; edits the partial grid returned by gen_obs_grid().
        - Returns:
            rgb_image: rendered RGB numpy array from the edited partial grid
            changed_caption: language caption for the edited partial view (via env.get_description)
            changed_obs_index: numpy array shape (view_size, view_size, 3) of indices usable by get_description
        - If change_desc cannot be applied (target not visible), returns (None, None, None).
        """

        # helpers
        TYPE_TO_CLASS = {"ball": Ball, "box": Box, "key": Key}
        VALID_TYPES = list(TYPE_TO_CLASS.keys())

        words = change_desc.lower().split()

        # distance qualifier
        if "closest" in words:
            rank = "closest"
        elif "farthest" in words:
            rank = "farthest"
        elif "middle" in words or "in-between" in words:
            rank = "middle"
        else:
            rank = "any"

        # parse "to new_color new_type"
        if "to" not in words:
            return None, None, None
        idx = words.index("to")
        try:
            new_color = words[idx + 1]
            new_type = words[idx + 2]
        except IndexError:
            return None, None, None

        if new_color not in COLOR_NAMES or new_type not in VALID_TYPES:
            return None, None, None

        # parse old color/type target in the text (first occurrence)
        old_color = None
        old_type = None
        for w in range(len(words) - 1):
            if words[w] in COLOR_NAMES and words[w + 1] in VALID_TYPES:
                old_color = words[w]
                old_type = words[w + 1]
                break
        if old_color is None or old_type is None:
            return None, None, None

        # Get partial grid and visibility mask
        grid, vis_mask = env.unwrapped.gen_obs_grid()
        # grid is a Grid object representing the partial view
        view_size = grid.width  # partial grid is square; width == height == agent_view_size
        agent_pos = (view_size // 2, view_size - 1)  # agent location in the partial grid coords

        # collect candidates in partial grid
        candidates = []  # list of ((x,y), dist)
        for x in range(view_size):
            for y in range(view_size):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                # obj.type and obj.color are strings like 'ball', 'green', etc.
                if (obj.type == old_type) and (obj.color == old_color):
                    dist = abs(x - agent_pos[0]) + abs(y - agent_pos[1])  # Manhattan in partial view
                    candidates.append(((x, y), dist))

        if not candidates:
            return None, None, None

        # sort by distance
        candidates.sort(key=lambda t: t[1])
        if rank == "closest":
            target_coord = candidates[0][0]
        elif rank == "farthest":
            target_coord = candidates[-1][0]
        elif rank == "middle":
            target_coord = candidates[1][0]
        else:
            # fallback: closest
            target_coord = candidates[0][0]

        tx, ty = target_coord

        # Create replacement object in the partial grid.
        # Use the class constructors so the object behaves correctly.
        new_obj = TYPE_TO_CLASS[new_type](new_color)

        # Replace in partial grid (this does NOT change the full env)
        # Grid.set exists in MiniGrid; use it to place the new object in the partial grid.
        grid.set(tx, ty, new_obj)

        # Render edited partial grid to RGB image
        # agent_pos passed relative to the partial grid (center-bottom)
        # use env.unwrapped.agent_dir to preserve agent facing when rendering
        rgb_img = grid.render(
            env.tile_size,
            agent_pos=(agent_pos[0], agent_pos[1]),
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        changed_obs = grid.encode()

        # Get a natural-language caption for the edited partial observation
        changed_caption = env.unwrapped.get_description({"image" : changed_obs})

        return rgb_img, changed_caption

    env = RandomObjectsEnv(max_steps=20, size=8, tile_size=16)
    env = RGBImgPartialObsWrapper(env, tile_size=16)

    total_training_data = 10000
    validation_data = 100
    No_changes = 0
    paired_data = []  # list of dicts: {"frame": , "caption": , "changed_caption": , "changed_frame": }

    obs, info = env.reset()
    base_frame = obs["image"]
    base_caption = info["description"]

    while No_changes < total_training_data + validation_data:
        frame = obs["image"]
        caption = info["description"]

        # Generate change description based on what is actually visible
        change_desc = generate_change_description(env)

        if change_desc:
            changed_frame, changed_caption = apply_change(env, change_desc, caption)
            if changed_frame is not None:
                paired_data.append({
                    "frame": frame,
                    "caption": caption,
                    "change_description": change_desc,
                    "changed_caption" : changed_caption,
                    "changed_frame": changed_frame,
                })
            No_changes += 1

        # Step environment
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

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
            
            # Save edited image
            img_edit = Image.fromarray(data["changed_frame"])
            edit_path = os.path.join(images_dir, f"edited_{i:06d}.png")
            img_edit.save(edit_path)
            
            # Write metadata with _file_name keys for image paths
            f.write(json.dumps({
                "input_image_file_name": f"images/original_{i:06d}.png",
                "edit_prompt": data["change_description"],
                "edited_image_file_name": f"images/edited_{i:06d}.png",
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
            
            # Save edited image
            img_edit = Image.fromarray(data["changed_frame"])
            edit_path = os.path.join(images_dir, f"edited_{i:06d}.png")
            img_edit.save(edit_path)
            
            # Write metadata with _file_name keys for image paths
            f.write(json.dumps({
                "input_image_file_name": f"images/original_{i:06d}.png",
                "edit_prompt": data["change_description"],
                "edited_image_file_name": f"images/edited_{i:06d}.png",
            }) + "\n")

    print(f"Saved {len(val_data)} paired validation examples to {save_dir}")