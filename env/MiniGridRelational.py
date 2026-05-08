import os
import math
import random
import numpy as np

import hydra
from PIL import Image
from omegaconf import DictConfig

from minigrid.core.constants import COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import (
    fill_coords,
    point_in_rect,
)
from minigrid.wrappers import RGBImgPartialObsWrapper, RGBImgObsWrapper, ImgObsWrapper

class RelationalBall(Ball):
    def __init__(self, color: str = "blue"):
        super().__init__(color)
        self.on_target_color = None

    def render(self, img):
        if self.on_target_color is not None:
            color = COLORS[self.on_target_color] / 2
            fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)
        super().render(img)

class Box(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains

    def can_pickup(self):
        return False

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return False

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Box is a fixed landmark and cannot be toggled/removed
        return False

class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color: str = "blue"):
        super().__init__("floor", color)

    def can_overlap(self):
        return False

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)

class RelationalPickPlaceEnv(MiniGridEnv):
    """
    MiniGrid environment for relational pick-and-place task.
    - task_mode == "source": Agent picks a blue ball and places it on the green floor target.
    - task_mode == "target": Agent picks a red ball and places it *beside* a yellow box.
    - task_mode == "target2": Agent picks a red ball and places it at the symmetric opposite of a yellow box.
    """

    def __init__(self, config: DictConfig, **kwargs):
        self.size = config.get("size", 8)
        self.task_mode = config.get("task_mode", "source")
        self.verbose = config.get("verbose", False)
        
        max_steps = config.get("max_steps", 20)
        
        def gen_mission():
            if self.task_mode == "source":
                return "Pick up the blue ball and drop it on the green target."
            elif self.task_mode == "target":
                return "Pick up the red ball and drop it beside the yellow box."
            else:
                return "Pick up the red ball and drop it at the symmetric opposite of the yellow box."

        mission_space = MissionSpace(mission_func=gen_mission)
        
        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            see_through_walls=False,
            max_steps=max_steps,
            highlight=False,
            **kwargs,
        )
        self.env_name = "MiniGridRelational"
        self.env_description = self._get_environment_description()
        self.reset_metrices()

    def reset_metrices(self):
        self.agent_performance = {
            "successful_pick": 0,
            "successful_drop": 0
        }
        
    def get_performance_metric(self):
        return self.agent_performance

    def _get_environment_description(self):
        """
        Returns a textual description of the environment dynamics, object affordances,
        agent capabilities, and scene variability.
        This text will be appended to the LLM prompt for imagination reasoning.
        """
        description = (
            "Environment context:\n"
            "- The agent operates in a fully observable 2D gridworld consisting of an 8×8 grid. "
            "- The grid is bordered by impassable wall tiles that occupy the entire outermost row and column on all four sides. "
            "- This leaves a 6×6 interior (columns 1–6, rows 1–6) as the usable play area where the agent, objects, and landmarks can be placed. "
            "- The agent has full visibility of the entire grid at all times.\n"
            "- At the start of each episode, the agent, tool objects (balls), and landmarks or target areas are randomly placed within the interior of the room.\n"
            "- The agent can perform the following actions: rotate left, rotate right, move forward one cell, pick up an object in the cell directly ahead, and drop the held object into the cell directly ahead.\n"
            "- The agent can carry only one object at a time. A picked-up object remains in the agent's inventory until explicitly dropped.\n"
            "- Objects in the environment include: colored balls (portable tools the agent can pick up and drop) and colored boxes.\n"
            "- Colored floor tiles mark target areas that the agent can walk over and drop objects onto.\n"
            "- Walls and empty floor cells cannot be interacted with. The agent cannot move through walls or occupied cells.\n"
        )
        return description

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.task_mode == "source":
            # Tool Block
            self.tool_block = RelationalBall(color="blue")
            self.place_obj(self.tool_block)
            
            # Target Area (Floor tile allows the agent to intrinsically step over it, but breaks generic drops without override)
            self.target_area = Floor(color="green")
            self.target_pos = self.place_obj(self.target_area)
            self.landmark = None
            self.landmark_pos = None
        else:
            # Both "target" and "target2" use a red ball + yellow box
            self.tool_block = RelationalBall(color="red")
            self.place_obj(self.tool_block)
            
            self.landmark = Box(color="yellow")
            self.landmark_pos = self.place_obj(self.landmark)
            self.target_area = None
            self.target_pos = None
            
        self.place_agent()
        if self.task_mode == "source":
            self.mission = "Pick up the blue ball and drop it on the green target."
        elif self.task_mode == "target1":
            self.mission = "Pick up the red ball and drop it beside the yellow box."
        else:
            self.mission = "Pick up the red ball and drop it at the symmetric opposite of the yellow box."

    def get_description(self, obs):
        dir_names = {0: "right", 1: "down", 2: "left", 3: "up"}
        
        parts = [f"Agent is at ({self.agent_pos[0]}, {self.agent_pos[1]}) facing {dir_names[self.agent_dir]}."]
        
        if self.carrying == self.tool_block:
            parts.append(f"Agent is carrying the {self.tool_block.color} {self.tool_block.type}.")
        elif getattr(self.tool_block, 'cur_pos', None) is not None:
            parts.append(f"The {self.tool_block.color} {self.tool_block.type} is at ({self.tool_block.cur_pos[0]}, {self.tool_block.cur_pos[1]}).")
            
        if self.task_mode == "source":
            if self.target_pos is not None:
                parts.append(f"The green target is at ({self.target_pos[0]}, {self.target_pos[1]}).")
        else:
            if self.landmark_pos is not None:
                # Assuming the landmark is never picked up since it's a Box and the agent drops tool blocks beside it
                parts.append(f"The yellow box is at ({self.landmark_pos[0]}, {self.landmark_pos[1]}).")
                
        return " ".join(parts)

    def step(self, action):
        carrying_before = self.carrying
        
        # Override Drop Action Logic to forcefully intercept on colored Floor tiles
        if action == self.actions.drop and self.carrying:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            # In source mode, we physically allow placing ON the green floor target despite collision
            if fwd_cell is not None and isinstance(fwd_cell, Floor) and fwd_cell.color == "green":
                if hasattr(self.carrying, 'on_target_color'):
                    self.carrying.on_target_color = fwd_cell.color
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        self.previous_state = self.obs
        obs, dict_reward, terminated, truncated, info = super().step(action)
        
        # Base penalty for time
        reward = 0.0

        # add intermediate reward for picking up the correct tool ball
        if action == self.actions.pickup and carrying_before is None and self.carrying is not None:
            if hasattr(self.carrying, 'on_target_color'):
                self.carrying.on_target_color = None
            if self.carrying == self.tool_block:
                reward += 1.0
                self.agent_performance["successful_pick"] += 1
                if self.verbose:
                    print(f"Intermediate Success: Picked up the {self.tool_block.color} ball!")
        
        # Reward shaping & termination logic evaluation!
        if action == self.actions.drop and carrying_before is not None and self.carrying is None:
            drop_pos = self.front_pos
            if self.task_mode == "source":
                if tuple(drop_pos) == tuple(self.target_pos):
                    reward += 10.0
                    terminated = True
                    self.agent_performance["successful_drop"] += 1
                    if self.verbose:
                        print("Success! Dropped exactly on green target area.")
                else:
                    terminated = True
                    if self.verbose:
                        print(f"Failed. Dropped at {drop_pos} instead of target {self.target_pos}.")
                        
            elif self.task_mode == "target1":
                # Compute all 8 surrounding cells of the landmark
                lx, ly = self.landmark_pos
                neighbors = [
                    (lx + dx, ly + dy)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if not (dx == 0 and dy == 0)
                ]
                # Filter out cells that are walls (outermost border)
                from minigrid.core.world_object import Wall
                valid_neighbors = [
                    pos for pos in neighbors
                    if 0 <= pos[0] < self.grid.width and 0 <= pos[1] < self.grid.height
                    and not isinstance(self.grid.get(*pos), Wall)
                ]
                if tuple(drop_pos) in valid_neighbors:
                    reward += 10.0
                    terminated = True
                    self.agent_performance["successful_drop"] += 1
                    if self.verbose:
                        print(f"Success! Dropped beside yellow box at {tuple(drop_pos)}.")
                else:
                    terminated = True
                    if self.verbose:
                        print(f"Failed. Dropped at {tuple(drop_pos)}, not adjacent to landmark at {tuple(self.landmark_pos)}.")

            elif self.task_mode == "target2":
                # Symmetric opposite of the landmark through the grid center
                lx, ly = self.landmark_pos
                sym_x = (self.grid.width - 1) - lx
                sym_y = (self.grid.height - 1) - ly
                if tuple(drop_pos) == (sym_x, sym_y):
                    reward += 10.0
                    terminated = True
                    self.agent_performance["successful_drop"] += 1
                    if self.verbose:
                        print(f"Success! Dropped at symmetric opposite ({sym_x}, {sym_y}) of yellow box at ({lx}, {ly}).")
                else:
                    terminated = True
                    if self.verbose:
                        print(f"Failed. Dropped at {tuple(drop_pos)}, expected symmetric opposite ({sym_x}, {sym_y}).")

        self.obs = obs
        info["description"] = self.get_description(obs)
            
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.obs = obs
        if self.verbose:
            info["description"] = self.get_description(obs)
        return obs, info

@hydra.main(version_base=None, config_path="../config/env", config_name="MiniGridRelational.yaml")
def main(args: DictConfig) -> None:
    env = RelationalPickPlaceEnv(args)
    env = RGBImgObsWrapper(env, tile_size=8) # Un-comment to force hard pixel pipeline instead of int arrays 
    env = ImgObsWrapper(env)
    for episode in range(500):
        obs, info = env.reset()
        if episode == 0:
            print("Mission:", env.unwrapped.mission)
        print(f"\n--- Episode {episode + 1} ---")
        print("Initial Info:", info.get("description", ""))
        
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
        print("Final Step Info:", info.get("description", ""))
        
    env.close()

if __name__ == "__main__":
    main() 
