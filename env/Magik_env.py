import time
import hydra
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from PIL import Image
from typing import Any, Callable
from omegaconf import DictConfig
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Ball, Wall
from minigrid.minigrid_env import MiniGridEnv
from scipy.spatial.distance import cityblock


class FlattenedFrameStack(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert hasattr(env, "num_stack"), \
            "FlattenedFrameStack must wrap an instance of FrameStack with 'num_stack' attribute."

        self.num_stack = env.num_stack  # Get num_stack from FrameStack
        old_shape = self.observation_space.shape  # (num_stack, H, W, C)
        assert len(old_shape) == 4, f"Expected stacked observation shape like (N, H, W, C), got {old_shape}"

        stack, height, width, channels = old_shape
        assert stack == self.num_stack, "Mismatch between num_stack and observation shape"

        new_shape = (stack * channels, height, width)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8
        )

    def observation(self, observation):
        """
        Converts (num_stack, H, W, C) → (num_stack * C, H, W)
        """
        obs = np.transpose(observation, (0, 3, 1, 2))  # (N, C, H, W)
        obs = obs.reshape(-1, *obs.shape[2:])          # (N*C, H, W)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.observation(obs)
        return obs, info
    
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


class MultiObjectMiniGridEnv(MiniGridEnv):
    """
    Custom MiniGrid Env that loads parameters from a config dictionary.
    """

    def __init__(self, config, **kwargs):
        # --- Extract parameters from the config dictionary ---
        self.size = config.get('size', 7)
        max_steps = config.get('max_steps', 4 * self.size**2)
        render_mode = config.get('render_mode', None)
        self.verbose = config.get('verbose', True)
        agent_view_size = config.get('agent_view_size', 7)
        self.verbose = config.get('verbose', False)

        # Get agent start config safely
        agent_start_config = config.get('agent_start', {})
        self.agent_start_pos = agent_start_config.get('pos', None)
        self.agent_start_dir = agent_start_config.get('dir', None)

        # Custom environment parameters
        self.reward_object = config.get('reward_object', 'Green_ball')
        self.is_single_object = config.get('is_single_object', False)
        self.to_be_terminated = config.get('to_be_terminated', False)
        self.see_through_walls = config.get('see_through_walls', True)
        self.wall_color = config.get('wall_color', 'grey')
        # --- End of config extraction ---

        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")
        self.max_step = max_steps

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            see_through_walls=self.see_through_walls,
            max_steps=max_steps,
            render_mode=render_mode,
            agent_view_size = agent_view_size,
            **kwargs,
        )

        # Initialize the agent's inventory as an empty list
        self.inventory = []
        # Define the maximum inventory size
        self.inventory_size = 2

        if self.is_single_object:
            self.env_name = "RedBallPickUpSingleObject"
        elif not self.is_single_object and self.reward_object == "Both":
            self.env_name = "BothPickUpMultipleObject"
        elif not self.is_single_object and self.reward_object == "Red_ball":
            self.env_name = "RedPickUpMultipleObject"
        else:
            self.env_name = "GreenBallPickUpMultipleObject"
        self.action_space = spaces.Discrete(4)
        self.mission = self._gen_mission()
        self.env_description = self._get_environment_description()

    # ... (the rest of your class methods remain exactly the same) ...
    # reset, _gen_grid, gen_obs_grid, gen_obs, _reward, flatten_obs,
    # get_object_name, get_class, step, get_unprocesed_obs, _pickup, _drop,
    # _gen_mission, and render_partial_view_from_features
    
    def _get_environment_description(self):
        """
        Returns a textual description of the environment dynamics, object affordances,
        agent capabilities, and scene variability.
        This text will be appended to the LLM prompt for imagination reasoning.
        """
        description = (
            "Environment context:\n"
            "- The agent operates in a partially observable gridworld-like room. The agent sees a portion of the room.\n"
            "- At the start of each episode, the agent may spawn in a room surrounded "
            "by either grey walls or blue walls.\n"
            "- The agent can perform the following actions: rotate left, rotate right, "
            "move forward, and pick up objects that are within reach.\n"
            "- The environment may contain objects such as balls, keys, boxes, and gates.\n"
            "- Balls and keys can both be picked up; gates can only be opened using keys "
            "of matching color.\n"
            "- Since, the observation is partial, the agent can explore the environment by moving around to find the oject to perform different tasks.\n"
            "- If the agent perform pick action in front of an object that can be picked up.\n "
            "- If either there is no object in front or the object cannot be picked up and action choosen is pick, the action has no effect.\n"
            "- Once one object is picked, the object dissapears from the scene and it is added to agent's inventory, which it can hold forever. This doesn't prevent picking another object later.\n"
            "- The agent can store multiple objects in it's inventory, up to a maximum of two objects"
            "- Non-interactive elements (walls, floor, background) cannot be acted upon.\n"
            "- The agent receives a reward upon successfully completing the Target task "
            "(for example, picking the specified object or opening the correct gate).\n"
            "- Each episode ends once the Target task is completed or a maximum step limit is reached."
        )
        return description
    
    def reset(self, seed=None, **kwargs):
        # Optionally, if you need to set the seed:
        if seed is not None:
            self.seed(seed)

        self.inventory = []
        # Pass seed and any other keyword arguments to the parent class reset.
        obs, info = super().reset(seed=seed, **kwargs)
        self.obs = obs['image']
        self.carrying = []
        self.picked = None
        self.episode_step_to_reward = 0
        info = {}
        if self.verbose:
            info["description"] = self.get_description(obs["image"])
        return obs, info

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = CustomGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height, Wall(self.wall_color))
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        # Place one green ball at a random position
        if self.is_single_object and self.reward_object == 'Green_ball':
            self.green_ball_loc = self.place_obj(Ball('green'), max_tries=100)

        # Place one red_ball at a random position
        if self.is_single_object and self.reward_object == 'Red_ball':
            self.red_ball_loc = self.place_obj(Ball('red'), max_tries=100)

        if not self.is_single_object:
            self.green_ball_loc = self.place_obj(Ball('green'), max_tries=100)
            self.red_ball_loc = self.place_obj(Ball('red'), max_tries=100)

        self.mission = self._gen_mission()
            
    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view

        return grid, vis_mask
    
    def gen_obs(self):
        # Return full-grid encoding as RGB-like 3D array
        grid, vis_mask = self.gen_obs_grid()
        img = grid.encode(vis_mask)
        obs = {"image": img, "direction": self.agent_dir, "mission": self.mission}

        # Represent the inventory as an array of object observations
        inventory_obs = np.zeros((self.inventory_size, 3), dtype='uint8')

        for idx, item in enumerate(self.inventory):
            if item is not None:
                # Encode the object as an integer array [type, color, state]
                inventory_obs[idx] = np.array(
                    [OBJECT_TO_IDX[item.type], COLOR_TO_IDX[item.color], 0], dtype='uint8'
                )

        obs['inventory'] = inventory_obs

        return obs

    def _reward(self):
        return 1
        # return 1 - 0.9 * (self.episode_step_to_reward / self.max_steps)

    def flatten_obs(self, obs=None):
        """
        Flatten the observation into a 1D array.
        """
        if obs is None:
            obs = self.obs
        # Flatten the image and inventory
        flat_obs = np.concatenate((obs['image'][:,:,:-1].flatten(), np.eye(4)[obs['direction']]))
        return flat_obs

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

        return final_desc


    def get_class(self, observation=None):

        if observation is None:
            observation = self.obs
        object, _ = self.generate_caption(observation)

        if 'red ball' in object and 'green ball' in object:
            probabilities = np.array([0,0,1,0])
        elif 'red ball' in object:
            probabilities = np.array([1,0,0,0])
        elif 'green ball' in object:
            probabilities = np.array([0,1,0,0])
        else:
            probabilities = np.array([0,0,0,1])

        return probabilities

    def step(self, action):
        self.step_count += 1
        self.episode_step_to_reward += 1
        green_collection_reward = 0
        red_collection_reward = 0
        self.picked = None
        terminated = False
        truncated = False
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Handle the pickup action
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if len(self.inventory) < self.inventory_size:  # Maximum inventory size
                    # Add the object to the inventory
                    self.inventory.append(fwd_cell)
                    self.carrying.append(fwd_cell)
                    self.carrying[-1].cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    if isinstance(fwd_cell, Ball) and fwd_cell.color == 'green':
                        green_collection_reward = self._reward()
                        self.picked = 'Green_ball'
                        # self.episode_step_to_reward = 0
                    elif isinstance(fwd_cell, Ball) and fwd_cell.color == 'red':
                        self.picked = 'Red_ball'
                        red_collection_reward = self._reward()
                        # self.episode_step_to_reward = 0

        # Drop an object
        elif action == self.actions.drop:
            pass

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")
        
        obs = self.gen_obs()
        self.obs = obs['image']
        
        if self.reward_object == "Green_ball":
            reward = green_collection_reward - red_collection_reward
        elif self.reward_object == "Red_ball":
            reward = red_collection_reward - green_collection_reward
        elif self.reward_object == "Both":
            reward = (green_collection_reward + red_collection_reward)/2
        else:
            raise ValueError("Reward object unidentified")

        if self.picked == self.reward_object and self.to_be_terminated:
            terminated = True
            
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()
        
        if len(self.inventory)>1:
            terminated = True
            
        info = {}
        if self.verbose:
            info["description"] = self.get_description(obs["image"])

        return obs, reward, terminated, truncated, info

    def get_unprocesed_obs(self):
        return self.obs

    def _pickup(self):
        """
        Handle the pickup action for multiple objects.
        """
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell and fwd_cell.can_pickup():
            if len(self.inventory) < self.inventory_size:  # Maximum inventory size
                # Add the object to the inventory
                self.inventory.append(fwd_cell)
                # Remove the object from the grid
                self.grid.set(*fwd_pos, None)
                if self.verbose:
                    print(f"Picked up {fwd_cell.type}-{fwd_cell.color}")
            else:
                if self.verbose:
                    print("Inventory full!")
        else:
            if self.verbose:
                pass
                # print("No object to pick up!")

    def _drop(self):
        """
        Handle the drop action for multiple objects.
        Dropping the last item in the inventory.
        """
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            if self.inventory:
                # Remove the last item from the inventory
                item = self.inventory.pop()
                # Place it on the grid in front of the agent
                item.cur_pos = fwd_pos
                self.grid.set(*fwd_pos, item)
                if self.verbose:
                    print(f"Dropped {item.type}-{item.color}")
            else:
                if self.verbose:
                    pass
                    # print("Inventory is empty!")
        else:
            if fwd_cell and self.inventory and self.verbose:
                print("Cannot drop here!")
    
    def _gen_mission(self):
        if self.is_single_object and self.reward_object == 'Green_ball':
            return f"Pick the green ball in the {self.wall_color} room"
        elif self.is_single_object and self.reward_object == 'Red_ball':
            return f"Pick the red ball in the {self.wall_color} room"
        elif not self.is_single_object and self.reward_object == 'Green_ball':
            return f"Pick green ball and avoid red ball in the {self.wall_color} room"
        elif not self.is_single_object and self.reward_object == 'Red_ball':
            return f"Pick red ball and avoid green ball in the {self.wall_color} room"
        else:
            return f"Pick red ball and green ball in the {self.wall_color} room"

    @staticmethod
    def render_partial_view_from_features(features, agent_dir, tile_size=32):
        if not isinstance(features, np.ndarray) or features.ndim != 3 or features.shape[2] != 3:
            raise ValueError("Features must be a NumPy array with shape (view_size, view_size, 3).")

        if agent_dir not in [0, 1, 2, 3]:
            raise ValueError("Invalid agent_dir. Must be 0 (right), 1 (down), 2 (left), or 3 (up).")

        view_size = features.shape[0]

        # Initialize the grid
        grid = Grid(view_size, view_size)

        # Agent's position remains the same (center of the last row)
        agent_pos = (view_size // 2, view_size - 1)  # (x, y) format

        # Agent's direction is now up (after transformation)
        agent_render_dir = 3  # Agent is facing "up"

        # Populate the grid
        for y in range(view_size):
            for x in range(view_size):
                obj_type_idx, color_idx, state_idx = features[x, y]
                
                color_idx = np.clip(color_idx, 0, 5)
                obj_type_idx = np.clip(obj_type_idx, 0, 10)
                state_idx = np.clip(state_idx, 0, 2)
                if obj_type_idx > 0:
                    obj = WorldObj.decode(type_idx=obj_type_idx, color_idx=color_idx, state=state_idx)
                    if obj:
                        grid.set(x, y, obj)
                    else:
                        grid.set(x, y, None)
                else:
                    grid.set(x, y, None)  # Unseen or empty

        # Render the grid with the agent's position and direction
        img = grid.render(
            tile_size=tile_size,
            agent_pos=agent_pos,
            agent_dir=agent_render_dir
        )
        return img
    
@hydra.main(version_base=None, config_path="../config/env", config_name="Magik_env")
def main(cfg: DictConfig) -> None:
    env = MultiObjectMiniGridEnv(cfg)
    from minigrid.wrappers import RGBImgPartialObsWrapper
    env = RGBImgPartialObsWrapper(env, tile_size=cfg.tile_size)
    from minigrid.wrappers import ImgObsWrapper
    env = ImgObsWrapper(env)
    for episode in range(3000):
        state, info = env.reset()
        done = False
        cumulative_reward = 0.0
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cumulative_reward += reward
            state = next_state

if __name__ == "__main__":
    main()