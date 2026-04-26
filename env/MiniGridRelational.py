import os
import math
import random
import numpy as np

import hydra
from PIL import Image
from omegaconf import DictConfig

from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Floor
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper, RGBImgObsWrapper, ImgObsWrapper

class RelationalPickPlaceEnv(MiniGridEnv):
    """
    MiniGrid environment for relational pick-and-place task.
    - task_mode == "source": Agent picks a blue ball and places it on a green floor target area.
    - task_mode == "target": Agent picks a purple ball and places it *beside* a yellow box (box).
    """

    def __init__(self, config: DictConfig, **kwargs):
        self.size = config.get("size", 8)
        self.task_mode = config.get("task_mode", "source")
        self.verbose = config.get("verbose", False)
        
        max_steps = config.get("max_steps", 150)
        
        def gen_mission():
            if self.task_mode == "source":
                return "Pick up the blue ball and drop it on the flat green target grid."
            else:
                return "Pick up the green ball and drop it near the grid where the yellow box is."

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
            "- The agent operates in a fully observable 2D gridworld of size 8x8 enclosed room with walls. The agent sees the entire room.\n"
            "- At the start of each episode, the agent, tool objects, and landmarks/target areas are randomly initialised in the environment.\n"
            "- The agent can perform the following actions: rotate left, rotate right, move forward, pick up objects, and drop objects. The drop action simply drops the carried object in the grid cell directly in front of the agent.\n"
            "- The environment contains objects such as balls (tools to pick up), boxes (landmarks like a 'box'), and colored floor tiles (target areas).\n"
            "- The agent's task is a relational pick-and-place task according to the mission string.\n"
            "- Depending on the task mode, the agent must either pick a specific object and drop it ON a target area, or drop it BESIDE (Nearest possible grid cell) a landmark grid cell.\n"
            "- The agent can carry only one object at a time. Once picked up, it is in the agent's inventory until dropped.\n"
            "- Non-interactive elements (walls, background) cannot be acted upon, but colored floor tiles can be walked over or dropped upon.\n"
            "- If an interactable object is present in the grid, the agent can't step onto that grid unless the agent picks it up.\n"
            "- The agent receives a reward upon successfully completing the target task (dropping the correct object at the correct relational location).\n"
            "- Each episode ends once the target task is completed or a maximum step limit is reached."
        )
        return description

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.task_mode == "source":
            # Tool Block
            self.tool_block = Ball(color="blue")
            self.place_obj(self.tool_block)
            
            # Target Area (Floor tile allows the agent to intrinsically step over it, but breaks generic drops without override)
            self.target_area = Floor(color="green")
            self.target_pos = self.place_obj(self.target_area)
            self.landmark = None
            self.landmark_pos = None
        else:
            self.tool_block = Ball(color="purple")
            self.place_obj(self.tool_block)
            
            # Landmark ("Yellow Duckie")
            self.landmark = Box(color="yellow")
            self.landmark_pos = self.place_obj(self.landmark)
            self.target_area = None
            self.target_pos = None
            
        self.place_agent()
        self.mission = "Pick up the blue ball and drop it on the green target." if self.task_mode == "source" else "Pick up the purple ball and drop it strictly beside the grid where the yellow box is."

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
            # In source mode, we physically allow placing ON the green floor despite collision
            if fwd_cell is not None and isinstance(fwd_cell, Floor) and fwd_cell.color == "green":
                self.carrying.cur_pos = np.array(fwd_pos)
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying = None

        self.previous_state = self.obs
        obs, dict_reward, terminated, truncated, info = super().step(action)
        
        # Base penalty for time
        reward = 0.0

        # add intermediate reward for picking the blue box (or ball)
        if action == self.actions.pickup and carrying_before is None and self.carrying is not None:
            if self.tool_block.color == "blue" and self.carrying == self.tool_block:
                reward += 1.0
                self.agent_performance["successful_pick"] += 1
                if self.verbose:
                    print("Intermediate Success: Picked up the blue ball!")
        
        # Reward shaping & termination logic evaluation!
        if action == self.actions.drop and carrying_before is not None and self.carrying is None:
            drop_pos = self.front_pos
            if self.task_mode == "source":
                if tuple(drop_pos) == tuple(self.target_pos):
                    reward += 10.0
                    terminated = True
                    self.agent_performance["successful_drop"] += 1
                    if self.verbose:
                        print("Success! Dropped exactly on green floor.")
                else:
                    terminated = True
                    if self.verbose:
                        print(f"Failed. Dropped at {drop_pos} instead of target {self.target_pos}.")
                        
            elif self.task_mode == "target":
                dist = abs(drop_pos[0] - self.landmark_pos[0]) + abs(drop_pos[1] - self.landmark_pos[1])
                if dist <= 2 and tuple(drop_pos) != tuple(self.landmark_pos):
                    reward += 10.0
                    terminated = True
                    self.agent_performance["successful_drop"] += 1
                    if self.verbose:
                        print(f"Success! Dropped beside yellow box at distance {dist}.")
                else:
                    terminated = True
                    if self.verbose:
                        print(f"Failed. Dropped at distance {dist} from landmark.")

        self.obs = obs
        info["description"] = self.get_description(obs)
        info["sensor_data"] = None
            
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.obs = obs
        if self.verbose:
            info["description"] = self.get_description(obs)
        info["sensor_data"] = None
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
