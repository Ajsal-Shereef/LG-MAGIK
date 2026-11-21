import minigrid
import gymnasium
import numpy as np

import sys
sys.path.append('.')
from env.env import SimplePickup

class MiniGridStateCaptioner:
    def __init__(self, env):
        self.directions = {
            0: "right",
            1: "down",
            2: "left",
            3: "up"
        }
        self.object_types = {
            'wall': "a wall",
            'floor': "an empty floor",
            'lava': "lava",
            'door': "a door",
            'key': "a key",
            'ball': "a ball",
            'box': "a box",
            'goal': "the goal",
            'agent': "the agent"
        }
        self.int_to_string = {
            1 : "first",
            2 : "second",
            3 : "third",
            4 : "fourth",
            5 : "fifth",
            6 : "sixth",
            7 : "secenth"
        }
        self.env = env

    def get_object_name(self, obj):
        if obj is None:
            return None
        obj_type = obj.type
        if hasattr(obj, 'color'):
            color = obj.color
            description = f"{color} {obj_type}"
            if obj.type == 'door':
                if obj.is_open:
                    description += " that is open"
                else:
                    description += " that is closed"
                if obj.is_locked:
                    description += " and locked"
                else:
                    description += " and unlocked"
            return description
        return obj_type

    def organize_view(self, obs):
        grid = np.transpose(obs['image'], (1, 0, 2))  # Transpose the grid to handle any direction uniformly

        # Get the field of view dimensions
        view_size = self.env.agent_view_size
        # Initialize the FOV view matrix
        fov = [[None for _ in range(view_size)] for _ in range(view_size)]
        # Fill the FOV based on agent's position
        for row in range(view_size):
            for col in range(view_size):
                obj_type, obj_colour, state = grid[row][col] 
                fov[row][col] = minigrid.core.world_object.WorldObj.decode(obj_type, obj_colour, state)

        return fov

    def generate_caption(self, obs):
        fov = self.organize_view(obs)
        agent_dir = obs['direction']
        direction_text = self.directions[agent_dir]

        caption = []
        caption.append(f"The agent is facing {direction_text}.")

        # List to hold descriptions for each row
        row_descriptions = []

        # Iterate through the rows of the FOV
        for row in range(5):
            # Initialize a list to store objects in this row
            objects_in_row = []

            left = fov[row][0]
            far_left = fov[row][1]
            right = fov[row][3]
            far_right = fov[row][4]

            # Gather descriptions for the current row
            if far_left:
                objects_in_row.append(f"far left: {self.get_object_name(far_left)}")
            if left:
                objects_in_row.append(f"left: {self.get_object_name(left)}")
            if right:
                objects_in_row.append(f"right: {self.get_object_name(right)}")
            if far_right:
                objects_in_row.append(f"far right: {self.get_object_name(far_right)}")

            # If there are visible objects in this row, create a description
            if objects_in_row:
                row_descriptions.append(f"In the {self.int_to_string[row + 1]} row, " + ", ".join(objects_in_row) + ".")

        # Combine the agent's direction with the row descriptions
        if row_descriptions:
            caption += row_descriptions
        else:
            caption.append("The agent sees nothing.")

        # Return the final caption
        return " ".join(caption)

# Create the MiniGrid environment with agent_view_size set to 5
# env = SimplePickup(max_steps = 500, render_mode='human')

# # Set the number of episodes and timesteps per episode
# num_episodes = 10
# max_timesteps = 50

# # Initialize the state captioner
# captioner = MiniGridStateCaptioner(env)

# for episode in range(num_episodes):
#     print(f"\nEpisode {episode + 1}/{num_episodes}")
    
#     obs = env.reset()

#     for t in range(max_timesteps):
#         env.render()

#         action = env.action_space.sample()

#         obs, reward, trunc, term, info = env.step(action)
#         done = trunc + term
#         caption = captioner.generate_caption(obs)
        
#         print(f"Timestep: {t + 1}, Reward: {reward}, Done: {done}")
#         print(f"State Description: {caption}")

#         if done:
#             print(f"Episode finished after {t + 1} timesteps.")
#             break

# env.close()
