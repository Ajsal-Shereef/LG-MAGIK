
import os
import random
import pygame
import mujoco
import numpy as np
import gymnasium as gym

from PIL import Image
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle

ARENA_LEN = 0.5  # From the XML file
ARM_LENGTH = 0.2  # From the XML file
# Target colors (Red, Green, Blue, Yellow)
TARGET_COLORS = [(255, 76, 76), (76, 255, 76), (76, 76, 255), (255, 255, 76)]

COLOUR_MAPPING = {0 : "Red",
                  1 : "Green",
                  2 : "Blue",
                  3 : "Yellow"}

CLASS_MAPPING = {"Red"   : 0,
                 "Green" : 1,
                 "Blue"  : 2,
                 "Yellow": 3}

# Define target positions in quadrants.
QUADRANT_MAPPING = {  (0.05, 0.35, 0.05, 0.35)     : 0,   # top-right
                      (-0.35, -0.05, 0.05, 0.35)   : 1,   # top-left
                      (-0.35, -0.05, -0.35, -0.05) : 2,   # bottom-left
                      (0.05, 0.35, -0.35, -0.05)   : 3,   # bottom-right
                    }

class Reacher4TargetsEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, goal_target_idx=0, max_steps=500, reaching_threshold=0.05, is_test = False, **kwargs):
        if not 0 <= goal_target_idx <= 3:
            raise ValueError("goal_target_idx must be between 0 and 3")
        
        EzPickle.__init__(self, goal_target_idx=goal_target_idx, max_steps=max_steps, **kwargs)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self,
            model_path=os.path.abspath("env/reacher.xml"),
            frame_skip=2,
            observation_space=observation_space,
            **kwargs
        )
        
        self.target_names = ["target0", "target1", "target2", "target3"]
        self.goal_target_idx = goal_target_idx
        self.max_steps = max_steps
        self.current_step = 0
        self.last_full_obs = None  
        self.screen = None  # Pygame screen, initialized only when needed
        self.fixed_target_positions = None
        self.reaching_threshold = reaching_threshold
        self.is_test = is_test
        self.obs_data = []
        self.obs_class = []
        self.env_name = f"{COLOUR_MAPPING.get(goal_target_idx)}_Reacher"

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.current_step += 1
    
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_positions = self.fixed_target_positions
        goal_pos = np.array([*target_positions[self.goal_target_idx], 0.01])
    
        # obs = self._get_obs()
        partial_obs = self._get_partial_obs()
        cls = self.get_class()[0]
        self.obs_data.append(partial_obs)
        self.obs_class.append(cls)

        if cls == self.goal_target_idx:
            distance = np.linalg.norm(fingertip_pos - goal_pos[:2])
            step_penalty = 0
        else:
            distance = 0.50
            step_penalty = 0

        reward_distance = -distance
        reward_control = -np.sum(np.square(action))
        reward = reward_distance + 0.10*reward_control + step_penalty
    
        truncated = self.current_step >= self.max_steps
        terminated = False
         
        if self.is_test:
            terminated = distance < self.reaching_threshold # Terminate if within threshold
    
        if self.render_mode == "human":
            self.last_full_obs = self.get_frame()
            self.render()
        
        return partial_obs, reward, terminated, truncated, dict(
            distance=distance,
            reward_distance=reward_distance,
            reward_control=reward_control
        )
    
    def is_reached(self):
        target_positions = self.fixed_target_positions
        target_diffs = [self.get_body_com("fingertip")[:2] - pos for pos in target_positions]
        distances = np.linalg.norm(target_diffs, axis=1)
        if distances[self.goal_target_idx] < self.reaching_threshold:
            return True
        else:
            return False
        
    
    def get_class(self):
        x,y = self.get_body_com("fingertip")[:2]
        # Check for origin
        if x == 0 and y == 0:
            quadrants = [0,1,2,3]
        # Check for points on the axes
        elif x == 0:
            if y>0:
                quadrants = [0,1]
        elif y == 0:
            if x>0:
                quadrants = [2,3]
        # Determine the quadrant based on signs of x and y
        elif x > 0 and y > 0:
            quadrants = [0]
        elif x < 0 and y > 0:
            quadrants = [1]
        elif x < 0 and y < 0:
            quadrants = [2]
        elif x > 0 and y < 0:
            quadrants = [3]
        quadrant = random.choice(quadrants)
        return CLASS_MAPPING.get(self.quadrant_info.get(quadrant)), quadrant

    def initialize_points_random(self, radius=0.40, n_points=4, min_dist=0.1, max_iter=1000):
        """
        Randomly sample n_points within a circle of given radius, ensuring that
        each point is at least min_dist away from the others.

        Parameters:
            radius (float): Maximum reach of the arm (radius of circle).
            n_points (int): Number of points to initialize.
            min_dist (float): Minimum allowed distance between any two points.
            max_iter (int): Maximum iterations to try before giving up.

        Returns:
            np.ndarray: An array of shape (n_points, 2) containing the (x, y) coordinates.
        """
        points = []
        iterations = 0

        while len(points) < n_points and iterations < max_iter:
            # Sample uniformly in polar coordinates: 
            theta = np.random.uniform(0, 2 * np.pi)
            # sqrt of uniform ensures uniform distribution in the circle
            r = np.sqrt(np.random.uniform(0, 1)) * radius
            candidate = np.array([r * np.cos(theta), r * np.sin(theta)])

            # Check if candidate is sufficiently far from all already selected points
            if all(np.linalg.norm(candidate - p) >= min_dist for p in points):
                points.append(candidate)
            iterations += 1

        if len(points) < n_points:
            raise ValueError("Could not find enough points with the given minimum distance. Try lowering min_dist.")

        return np.array(points)

    def reset_model(self):
        """Reset the environment state and return a partial image observation."""
        self.current_step = 0
        qpos = self.np_random.uniform(low=-0.37, high=0.37, size=self.model.nq) + self.init_qpos
        qvel = self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-8:] = 0  # Zero target velocities

        self.quadrant_info = {}
        available_quadrants = list(QUADRANT_MAPPING.keys()).copy()
        self.fixed_target_positions = []

        for i in range(4):
            q_idx = self.np_random.integers(low=0, high=len(available_quadrants))
            selected_quadrant = available_quadrants.pop(q_idx)
            x_low, x_high, y_low, y_high = selected_quadrant
            target_pos = np.array([
                self.np_random.uniform(low=x_low, high=x_high),
                self.np_random.uniform(low=y_low, high=y_high)
            ])
            self.fixed_target_positions.append(target_pos)
            qpos[2 + 2 * i: 2 + 2 * i + 2] = target_pos

            self.quadrant_info[QUADRANT_MAPPING.get(selected_quadrant)] = COLOUR_MAPPING.get(i)

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)  # Propagate the state
        obs = self._get_partial_obs()
        return obs
    
    @staticmethod
    def scale_angular_velocity(omega):
        return np.tanh(0.06 * omega)
    
    def sort_coordinates_by_quadrant(self, coordinates):
        quadrants = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

        for x, y in coordinates:
            if x > 0 and y > 0:
                quadrants["Q1"].append((x, y))
            elif x < 0 and y > 0:
                quadrants["Q2"].append((x, y))
            elif x < 0 and y < 0:
                quadrants["Q3"].append((x, y))
            elif x > 0 and y < 0:
                quadrants["Q4"].append((x, y))

        # Flatten into the correct order: Q1, Q2, Q3, Q4
        ordered_points = quadrants["Q1"] + quadrants["Q2"] + quadrants["Q3"] + quadrants["Q4"]
        return ordered_points

    def _get_partial_obs(self):
        theta = self.data.qpos.flat[:2]
        fingertip_pos = self.get_body_com("fingertip")[:2]
        sorted_cordinate = self.sort_coordinates_by_quadrant(self.fixed_target_positions)
        quadrant = self.get_class()[1]
        target_cordinate = sorted_cordinate[quadrant]
        target_diffs = [fingertip_pos - pos for pos in sorted_cordinate]
        distances = np.linalg.norm(target_diffs, axis=1)
        distance = distances[quadrant]
        colour = np.eye(4)[CLASS_MAPPING.get(self.quadrant_info[quadrant])]
        return np.concatenate([
            np.cos(theta),            #2   
            np.sin(theta),            #2
            target_cordinate,         #2
            self.scale_angular_velocity(self.data.qvel.flat[:2]),  #2
            fingertip_pos,            #2
            # target_diffs
            np.array([distance]),     #1
            colour        #4
        ]).astype(np.float32)


    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_positions = self.fixed_target_positions
        target_diffs = [fingertip_pos - pos for pos in target_positions]
        target_positions = np.concatenate(target_positions)
        distances = np.linalg.norm(target_diffs, axis=1)
        # target_diffs = np.concatenate(target_diffs)

        return np.concatenate([
            np.cos(theta),            #2   
            np.sin(theta),            #2
            target_positions,         #8
            self.scale_angular_velocity(self.data.qvel.flat[:2]),  #2
            fingertip_pos,
            # target_diffs
            distances                 #4
        ])
    
    def _make_frame(self, side, feature_obs=None):
        scaling = side / (2 * ARENA_LEN)
        base_pose = np.array([side // 2, side // 2])
        arm_length = ARM_LENGTH * scaling
        surface = pygame.Surface((side, side))
        surface.fill((76, 76, 76))

        # Use the current feature observation.
        if feature_obs is None:
            feature_obs = self._get_obs()
        cos_theta1, sin_theta1 = feature_obs[0], feature_obs[2]
        cos_theta2_rel, sin_theta2_rel = feature_obs[1], feature_obs[3]
        target_positions = feature_obs[4:12].reshape(4, 2)
        target_positions[:, 1] *= -1

        theta1 = -np.arctan2(sin_theta1, cos_theta1)
        theta2 = theta1 - np.arctan2(sin_theta2_rel, cos_theta2_rel)

        joint1 = base_pose + arm_length * np.array([np.cos(theta1), np.sin(theta1)])
        joint2 = joint1 + arm_length * np.array([np.cos(theta2), np.sin(theta2)])

        pygame.draw.line(surface, (229, 90, 153), base_pose, joint1, max(int(side / 100), 2))
        pygame.draw.line(surface, (229, 90, 153), joint1, joint2, max(int(side / 100), 2))
        pygame.draw.circle(surface, (50, 50, 50), base_pose, max(int(side / 62), 2))
        pygame.draw.circle(surface, (50, 50, 50), joint1, max(int(side / 62), 2))

        # Draw targets.
        for i, target in enumerate(target_positions):
            target_pos = (base_pose + target * scaling).astype(int)
            pygame.draw.circle(surface, TARGET_COLORS[i], target_pos, max(int(side / 62), 2))

        # Draw fingertip.
        fingertip_pos = self.get_body_com("fingertip")[:2].copy()
        fingertip_pos[1] *= -1
        fingertip_pixel = (base_pose + fingertip_pos * scaling).astype(int)
        pygame.draw.circle(surface, (50, 50, 50), fingertip_pixel, max(int(side / 62), 2))

        # Convert to numpy array.
        image = pygame.surfarray.array3d(surface)
        image = np.transpose(image, (1, 0, 2))
        return image
    
    def get_frame(self, hq_side = 512, feature_obs=None):
        """
        Render a high-quality full scene (512×512×3).

        Returns:
            np.array: High-resolution image of the full scene.
        """
        return self._make_frame(hq_side, feature_obs)
    
    def render(self, mode="human"):
        """Render the environment.

        Args:
            mode (str): The mode to render the environment. Options are "human", "rgb_array", and "depth_array".
        """
        self.render_mode = mode
        if self.render_mode == "human":
            return None
        elif self.render_mode == "rgb_array":
            return self.get_frame(512)
        elif self.render_mode == "depth_array":
            return self.get_frame(512)
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")


# Example usage
# if __name__ == "__main__":
#     env = Reacher4TargetsEnv(goal_target_idx=2, render_mode="human")
#     # env = gym.make("Reacher-v5", render_mode="human")
#     for _ in range(100):
#         obs, info = env.reset()
#         # print(f"Goal target: {env.target_names[env.goal_target_idx]}")
#         done = False
#         while not done:
#             action = env.action_space.sample()
#             obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated + truncated
#             # print(f"Step: {env.current_step}, Reward: {reward}, Distance: {info['distance']}")
#             if terminated or truncated:
#                 print("Episode ended")
#             if done:
#                 break
#     env.close()