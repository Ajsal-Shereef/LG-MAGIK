import hydra
import random
import numpy as np

from PIL import Image
from collections import deque
from typing import Any, Callable
from omegaconf import DictConfig
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.spaces import Box
from minigrid.core.world_object import WorldObj, Wall, Door, Goal, Key, Ball, Box as MGBox, Lava
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_NAMES, COLOR_TO_IDX, STATE_TO_IDX

class CustomGrid(Grid):
    def __init__(self, width: int, height: int):
        super(CustomGrid, self).__init__(width, height)
        
    def horz_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type)

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall,
    ):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type)
        
    def wall_rect(self, x: int, y: int, w: int, h: int, obj_type):
        self.horz_wall(x, y, w, obj_type)
        self.horz_wall(x, y + h - 1, w, obj_type)
        self.vert_wall(x, y, h, obj_type)
        self.vert_wall(x + w - 1, y, h, obj_type)

class CustomMiniGridEnv(MiniGridEnv):
    def __init__(self, config):
        self.config = config
        self.size = self.config.get('size', 10)
        self.max_steps = self.config.get('max_steps', 50)
        self.wall_color_options = self.config.get('wall_color_options', ["grey", "blue"])
        self.max_objects = self.config.get('max_objects', 4)
        self.agent_view_size = self.config.get('agent_view_size', 5)
        self.see_through_walls = self.config.get('see_through_walls', False)

        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")
        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            max_steps=self.max_steps,
            agent_view_size=self.agent_view_size,
            see_through_walls=self.see_through_walls,
        )
        # Fully observable RGB-encoded grid
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.size, self.size, 3),
            dtype=np.uint8
        )

    def _gen_grid(self, width, height):
        # Reset grid and draw outer border walls
        self.grid = CustomGrid(width, height)
        # Choose wall color: from options
        self.wall_color = random.choice(self.wall_color_options)
        self.grid.wall_rect(0, 0, width, height, obj_type=Wall(self.wall_color))

        # Randomly choose layout: no wall, or one full middle wall (horizontal/vertical)
        layout_choice = random.choice(["none", "horizontal", "vertical"])

        door_color_used = None
        wall_info = None  # ("horizontal", y) or ("vertical", x)
        door_pos = None
        
        def build_full_wall_with_opening(orientation):
            nonlocal door_color_used, wall_info, door_pos

            if orientation == "horizontal":
                # Choose an interior row for a full wall
                wy = random.randint(2, height - 3)
                for x in range(1, width - 1):
                    self.grid.set(x, wy, Wall(self.wall_color))
                # Choose an interior opening and type
                ox = random.randint(2, width - 3)
                opening_choice = random.choice(["gap", "door_unlocked", "door_locked"])
                if opening_choice == "gap":
                    self.grid.set(ox, wy, None)
                else:
                    dcolor = random.choice(list(COLOR_NAMES))
                    locked = opening_choice == "door_locked"
                    d = Door(dcolor, is_locked=locked)
                    self.grid.set(ox, wy, d)
                    door_color_used = dcolor
                    door_pos = (ox, wy)
                wall_info = ("horizontal", wy)

            else:  # vertical
                wx = random.randint(2, width - 3)
                for y in range(1, height - 1):
                    self.grid.set(wx, y, Wall(self.wall_color))
                oy = random.randint(2, height - 3)
                opening_choice = random.choice(["gap", "door_unlocked", "door_locked"])
                if opening_choice == "gap":
                    self.grid.set(wx, oy, None)
                else:
                    dcolor = random.choice(list(COLOR_NAMES))
                    locked = opening_choice == "door_locked"
                    d = Door(dcolor, is_locked=locked)
                    self.grid.set(wx, oy, d)
                    door_color_used = dcolor
                    door_pos = (wx, oy)
                wall_info = ("vertical", wx)

        if layout_choice in ("horizontal", "vertical"):
            build_full_wall_with_opening(layout_choice)

        # Place the agent somewhere in the top-left half, excluding door position if exists
        def agent_reject_fn(env, pos):
            if door_pos is None:
                return False
            return tuple(pos) == tuple(door_pos)

        self.place_agent(top=(1, 1), size=(max(1, width // 2), max(1, height // 2)))

        # Randomize goal location anywhere in the interior, not on doors/walls/agent
        def reject_for_goal(env, pos):
            return (door_pos is not None and tuple(pos) == tuple(door_pos)) or tuple(pos) == tuple(env.agent_pos)

        goal_pos = self.place_obj(
            Goal(),
            top=(1, 1),
            size=(width - 2, height - 2),
            reject_fn=reject_for_goal
        )

        # Common reject function for other objects
        def reject_on_agent_door(env, pos):
            if tuple(pos) == tuple(env.agent_pos):
                return True
            if tuple(pos) == tuple(goal_pos):
                return True
            if door_pos is not None and tuple(pos) == tuple(door_pos):
                return True
            return False

        # Ensure at most 4 objects total; if a colored door was used, include a same-color key
        MAX_OBJECTS = self.max_objects
        objs_to_place = []

        # If any colored door was used (locked or unlocked), place a matching key
        required_key = None
        if door_color_used is not None:
            required_key = Key(door_color_used)
            objs_to_place.append(required_key)

        # Fill remaining slots (up to MAX_OBJECTS - 1 for lava) with random objects (Ball/Box/Key) in random colors
        remaining = MAX_OBJECTS - len(objs_to_place) - 1  # Reserve 1 for lava
        if remaining > 0:
            colors = list(COLOR_NAMES)
            random.shuffle(colors)
            types = [Ball, MGBox, Key]
            # Build a small candidate pool
            candidates = []
            for c in colors[:8]:
                for T in types:
                    # Avoid duplicating the required key color excessively (not strictly necessary)
                    if required_key is not None and T is Key and c == door_color_used:
                        continue
                    candidates.append(T(c))
            random.shuffle(candidates)
            objs_to_place.extend(candidates[:remaining])

        # Placement strategy:
        # - If there is a required key and a single middle wall, restrict the key to the agent's side.
        # - All other objects can go anywhere in the interior, excluding agent/goal/door cells.
        placed = 0
        for obj in objs_to_place:
            if obj is required_key and wall_info is not None:
                if wall_info[0] == "vertical":
                    wx = wall_info[1]
                    # Agent starts on the left/top; bias key to the left side of the vertical wall
                    key_top = (1, 1)
                    key_size = (max(1, wx - 1), height - 2)
                else:
                    wy = wall_info[1]
                    # Bias key above the horizontal wall
                    key_top = (1, 1)
                    key_size = (width - 2, max(1, wy - 1))
                # Fall back to full interior if the computed region is degenerate
                if key_size[0] <= 0 or key_size[1] <= 0:
                    key_top = (1, 1)
                    key_size = (width - 2, height - 2)
                self.place_obj(obj, top=key_top, size=key_size, reject_fn=reject_on_agent_door)
            else:
                self.place_obj(obj, top=(1, 1), size=(width - 2, height - 2), reject_fn=reject_on_agent_door)
            placed += 1
            if placed >= MAX_OBJECTS:
                break

        # Place some lava (1 cell) if total objects < MAX_OBJECTS
        if placed < MAX_OBJECTS:
            # Place lava anywhere in the interior, excluding agent/goal/door cells
            lava_placed = False
            while not lava_placed:
                lava_pos = self.place_obj(Lava(), top=(1, 1), size=(width - 2, height - 2), reject_fn=reject_on_agent_door)
                # Check if path to goal still exists after placing lava
                if self.is_reachable(tuple(self.agent_pos), tuple(goal_pos)):
                    lava_placed = True
                else:
                    # Remove if it blocks
                    self.grid.set(lava_pos[0], lava_pos[1], None)

        # Store instance variables for description
        self.door_pos = door_pos
        self.wall_info = wall_info
        self.goal_pos = goal_pos

        # Mission
        self.mission = "Reach the goal"

        
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
        image = observation
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
        if self.wall_color == "blue" and saw_wall:
            initial_description = "Agent is in a room surrounded by blue walls."
        elif self.wall_color == "grey" and saw_wall:
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

        # Check for obstruction
        obstruction = self.find_obstruction()
        if obstruction:
            door_obj = self.grid.get(*self.door_pos)
            if door_obj:
                obstruction_str = f" The {obstruction.color} {obstruction.type} is obstructing the {door_obj.color} door."
                final_desc += obstruction_str

        return final_desc

    def find_obstruction(self):
        if self.door_pos is None or self.wall_info is None:
            return None
        door_x, door_y = self.door_pos
        orient, pos = self.wall_info

        # Define agent side
        agent_pos = tuple(self.agent_pos)
        if orient == "vertical":
            agent_side = "left" if agent_pos[0] < pos else "right"
            check_cells = []
            check_cells.append((door_x - 1, door_y) if agent_side == "left" else (door_x + 1, door_y))
        else:
            agent_side = "top" if agent_pos[1] < pos else "bottom"
            check_cells = []
            check_cells.append((door_x, door_y - 1) if agent_side == "top" else (door_x, door_y + 1))

        obstruction = None
        for x, y in check_cells:
            obj = self.grid.get(x, y)
            if obj is not None and obj.type not in {"empty", "goal", "door"}:
                obstruction = obj
                break
        return obstruction

    def is_reachable(self, start, goal):
        queue = deque([start])
        visited = set([start])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    obj = self.grid.get(nx, ny)
                    if obj is None or obj.type in {"key", "ball", "box", "goal", "door"}:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def gen_obs(self):
        # Return full-grid encoding as RGB-like 3D array
        grid, _ = self.gen_obs_grid()
        img = grid.encode()
        return img

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Custom sparse reward for reaching the goal
        current_cell = self.grid.get(*self.agent_pos)
        if current_cell and current_cell.type == "goal":
            reward = 1.0
            terminated = True
        description = self.get_description(obs)
        return obs, reward, terminated, truncated, {"description" : description}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        description = self.get_description(obs)
        return self.gen_obs(), {"description" : description}


@hydra.main(version_base=None, config_path="../config/env", config_name="Pretraining")
def main(cfg: DictConfig) -> None:
    env = CustomMiniGridEnv(cfg)
    for episode in range(3000):
        state, info = env.reset()
        done = False
        cumulative_reward = 0.0
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cumulative_reward += reward
            state = next_state

if __name__ == "__main__":
    main()