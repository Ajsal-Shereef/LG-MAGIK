import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, DIR_TO_VEC, IDX_TO_OBJECT, IDX_TO_COLOR
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Door, Goal, Key, Wall, Ball
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



class MultiObjectMiniGridEnv(MiniGridEnv):
    """
    Custom MiniGrid Env.
    """

    def __init__(
        self,
        max_steps,
        size=7,
        render_mode = None,
        agent_start_pos=None,
        agent_start_dir=None,
        verbose=True,
        reward_object = 'Green_ball',
        is_single_object = False,
        to_be_terminated = False,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.reward_object = reward_object
        self.is_single_object = is_single_object

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2
        self.max_step = max_steps
        self.verbose = verbose

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            render_mode = render_mode,
            **kwargs,
        )

        # Initialize the agent's inventory as an empty list
        self.inventory = []

        # Define the maximum inventory size
        self.inventory_size = 2
        self.to_be_terminated = to_be_terminated
        # self.action_space = spaces.Discrete(4)
        
        if self.is_single_object:
            self.env_name = "RedBallPickUpSingleObject"
        elif not self.is_single_object and self.reward_object == "Both":
            self.env_name = "BothPickUpMultipleObject"
        elif not self.is_single_object and self.reward_object == "Red_ball":
            self.env_name = "RedPickUpMultipleObject"
        else:
            self.env_name = "GreenBallPickUpMultipleObject"


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
        return obs, info

    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # # # Place a ball square in the bottom-right corner
        # self.grid.set(4, 2, Ball(color='green'))
        # self.green_ball_loc = (4,2)
        
        # # # # Place a ball square in the bottom-right corner
        # self.grid.set(2, 4, Ball(color='red'))
        # self.red_ball_loc = (2,4)
        
        # Place one green ball at a random position
        if self.is_single_object and self.reward_object == 'Green_ball':
            self.green_ball_loc = self.place_obj(Ball('green', can_overlap=True), max_tries=100)

        # Place one red_ball at a random position
        if self.is_single_object and self.reward_object == 'Red_ball':
            self.red_ball_loc = self.place_obj(Ball('red', can_overlap=True), max_tries=100)

        if not self.is_single_object:
            self.green_ball_loc = self.place_obj(Ball('green', can_overlap=True), max_tries=100)
            self.red_ball_loc = self.place_obj(Ball('red', can_overlap=True), max_tries=100)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Pick the green ball"

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
        """
        Generate the agent's view including the inventory.
        """
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

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

    def generate_caption(self, observation):
        """
        Generate a natural language caption for the MiniGrid observation.
        Parameters:
            observation (dict): The observation dictionary with keys 'image', 'direction', 'mission'.
        Returns:
            str: A caption describing the state.
        """
        # Extract relevant data from observation
        image = observation
        view_size = image.shape[0]
        agent_position = (view_size // 2, view_size - 1)
        # Initialize lists to store object descriptions and distances
        descriptions = []
        distances = []
        # Iterate over the field of view and collect descriptions for each visible object
        for i in range(view_size):
            for j in range(view_size):
                # Extract object type, color, and state from the image
                obj_type, obj_color, obj_state = image[i, j]
                # Skip "wall" and "empty" objects based on their indices
                if obj_type in {OBJECT_TO_IDX['wall'], OBJECT_TO_IDX['empty']}:
                    continue
                
                # Generate a natural language description of the object
                obj_description = self.get_object_name(obj_type, obj_color, obj_state)
                # Calculate the Euclidean distance from the agent to this object
                distance = np.sqrt((i - agent_position[0])**2 + (j - agent_position[1])**2)
                # Store the object description and distance
                descriptions.append(obj_description)
                distances.append((distance, obj_description))
        # Generate the final caption based on the collected descriptions
        if not descriptions:
            # No objects detected in the view
            return {}, "You see nothing."
        # Sort the objects by proximity and identify the closest one
        distances.sort()
        min_distance = distances[0][0]
        closest_objects = [desc for dist, desc in distances if dist == min_distance]
        # Join all objects descriptions
        objects_seen = ", ".join(descriptions)
        # Create the caption
        # Create the caption mentioning all closest objects
        if len(closest_objects) == 1:
            caption = f"You see {objects_seen} and you are close to {closest_objects[0]}."
        else:
            closest_desc = " and ".join(closest_objects)
            caption = f"You see {objects_seen} and you are close to {closest_desc}."
        return objects_seen, caption


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
            # if not fwd_cell and self.carrying:
            #     if self.inventory:
            #         # Remove the last item from the inventory
            #         item = self.inventory.pop(0)
            #     self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying[0])
            #     self.carrying[0].cur_pos = fwd_pos
            #     self.carrying.pop(0)
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
            reward = green_collection_reward
            # cls = self.get_class(self.obs)
            # if np.array_equal(cls, np.array([0,1,0,0])) or np.array_equal(cls, np.array([0,0,1,0])) and not np.array_equal(self.green_ball_loc, self.agent_pos):
            #     reward += 1/(1+cityblock(self.green_ball_loc, self.agent_pos))
        elif self.reward_object == "Red_ball":
            reward = red_collection_reward
            # cls = self.get_class(self.obs)
            # if np.array_equal(cls, np.array([1,0,0,0])) or np.array_equal(cls, np.array([0,0,1,0])) and not np.array_equal(self.green_ball_loc, self.agent_pos):
            #     reward += 1/(1+cityblock(self.red_ball_loc, self.agent_pos))
        else:
            reward = (green_collection_reward + red_collection_reward)/2

        if self.picked == self.reward_object and self.to_be_terminated:
            terminated = True
            
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()
        
        if len(self.inventory)>1:
            terminated = True

        return obs, reward, terminated, truncated, {}

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
    
    @staticmethod
    def _gen_mission():
        return "Pick the green ball"
    
    @staticmethod
    def render_partial_view_from_features(features, agent_dir, tile_size=32):
        """
        Render an image-based observation from a feature-based observation,
        transforming it so the agent is always facing up.

        Args:
            features: Feature-based observation (view_size x view_size x 3).
                      Contains [object type, color, state] for each grid cell.
            agent_dir: Integer representing the agent's direction 
                       (0: right, 1: down, 2: left, 3: up).
            tile_size: Size of each tile in the rendered image.

        Returns:
            Rendered image as an RGB array.
        """
        # Validate input
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