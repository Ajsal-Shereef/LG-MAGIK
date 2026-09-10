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

class Box(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains

    def can_pickup(self):
        return False

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return True

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
        return self.color != "green"

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)

class RelationalPickPlaceEnv(MiniGridEnv):
    """
    MiniGrid environment for relational pick-and-place task.
    - task_mode == "source":  Agent picks a blue ball and places it on the green floor target.
    - task_mode == "target1": Agent picks a red ball and places it *adjacent to* a yellow box.
    - task_mode == "target2": Agent picks a red ball and places it at the symmetric opposite of
                              a yellow box.
    - task_mode == "target3": Agent picks a red ball and places it on a cell that is exactly
                              Manhattan-distance 2 from the yellow box, choosing the one nearest
                              (Manhattan) to the agent's current position not colluding with the agent location.
    - task_mode == "target4": Agent picks a red ball and places it at the symmetric opposite (through
                              the grid centre) of the cell two cells above the yellow box.
                              The yellow box is constrained at episode generation to by >= 3 so
                              the formula always lands in the playable interior.
    - task_mode == "target5": Two-pair pickup-drop task. Two coloured balls (red, purple) and two
                              coloured floor tiles (yellow, grey). Pairing is fixed:
                              red ball -> yellow target, purple ball -> grey target.
                              The agent must complete BOTH pairs; either pair may be done
                              first. Wrong drops end the episode.
                              Max return per episode = +1 (pick A) + +10 (drop A) +
                              +1 (pick B) + +10 (drop B) = 22.
    """

    def gen_mission(self):
        if self.task_mode == "source":
            return "Pick up the blue ball and drop it on the green target."
        elif self.task_mode == "target1":
            return "Pick up the red ball and drop it strctly on a cell immediately adjacent to the yellow box."
        elif self.task_mode == "target2":
            return "Pick up the red ball and drop it at the symmetric opposite of the yellow box."
        elif self.task_mode == "target3":
            return "Pick up the red ball and drop it on a cell that is exactly 2 Manhattan-distance away from the yellow box, choosing the one nearest to the agent not colluding with the agent location (choose only one if multiple)."
        elif self.task_mode == "target5":
            return (
                "Pick up the red ball and drop it on the yellow target and "
                "pick up the purple ball and drop it on the grey target. "
                "The two pairs may be completed in any order."
            )
        else:
            return (
                "Pick up the red ball and drop it at the symmetric opposite "
                "of the cell two cells above the yellow box."
            )

    def __init__(self, config: DictConfig, **kwargs):
        self.size = config.get("size", 8)
        self.task_mode = config.get("task_mode", "source")
        self.verbose = config.get("verbose", False)
        
        max_steps = config.get("max_steps", 20)
        
        mission_space = MissionSpace(mission_func=lambda: self.gen_mission())
        
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
            "- When a ball is correctly dropped on its matching coloured floor target, both the ball and the floor tile are removed from the grid (the cell becomes empty grid).\n"
            "- Walls and empty floor cells cannot be interacted with. The agent cannot move through walls or occupied cells.\n"
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
        elif self.task_mode == "target5":
            # Two-step sequential task. Two balls + two floor targets.
            # Pairing: red -> yellow, purple -> grey.
            self.ball_a = Ball(color="red")
            self.ball_a_pos = self.place_obj(self.ball_a)
            self.ball_b = Ball(color="purple")
            self.ball_b_pos = self.place_obj(self.ball_b)

            self.target_a = Floor(color="yellow")
            self.target_a_pos = self.place_obj(self.target_a)
            self.target_b = Floor(color="grey")
            self.target_b_pos = self.place_obj(self.target_b)

            # Per-episode bookkeeping for the two subgoals.
            self._target5_done = {"a": False, "b": False}

            # Compat with code paths that read these attributes.
            self.tool_block = self.ball_a
            self.landmark = None
            self.landmark_pos = None
            self.target_area = None
            self.target_pos = None
        else:
            # All red-ball / yellow-box targets share the same world layout.
            self.landmark = Box(color="yellow")
            if self.task_mode == "target4":
                # The drop cell for target4 is sym((bx, by - 2)) =
                # (W-1-bx, W-1-(by-2)) = (W-1-bx, W+1-by). For 8x8 that
                # is (7-bx, 9-by). It only stays inside the playable
                # interior (1..6) when by >= 3, so reject any placement
                # that would make the task impossible.
                self.landmark_pos = self.place_obj(
                    self.landmark,
                    reject_fn=lambda env, pos: pos[1] < 3,
                )
            else:
                self.landmark_pos = self.place_obj(self.landmark)

            self.tool_block = Ball(color="red")
            if self.task_mode == "target1":
                self.place_obj(
                    self.tool_block,
                    reject_fn=lambda env, pos: abs(pos[0] - self.landmark_pos[0]) + abs(pos[1] - self.landmark_pos[1]) <= 1,
                )
            elif self.task_mode == "target2":
                self.place_obj(
                    self.tool_block,
                    reject_fn=lambda env, pos: pos == (env.grid.width - 1 - self.landmark_pos[0], env.grid.height - 1 - self.landmark_pos[1]),
                )
            elif self.task_mode == "target3":
                self.place_obj(
                    self.tool_block,
                    reject_fn=lambda env, pos: abs(pos[0] - self.landmark_pos[0]) + abs(pos[1] - self.landmark_pos[1]) == 2,
                )
            elif self.task_mode == "target4":
                self.place_obj(
                    self.tool_block,
                    reject_fn=lambda env, pos: pos == (env.grid.width - 1 - self.landmark_pos[0], env.grid.height - 1 - (self.landmark_pos[1] - 2)),
                )
            else:
                self.place_obj(self.tool_block)

            self.target_area = None
            self.target_pos = None
            
        self.place_agent()
        if self.task_mode in ["target1", "target2", "target3", "target4", "target5"]:
            while True:
                pos = self.agent_pos
                invalid = False
                if self.task_mode == "target1":
                    invalid = abs(pos[0] - self.landmark_pos[0]) + abs(pos[1] - self.landmark_pos[1]) <= 1
                elif self.task_mode == "target2":
                    invalid = pos == (self.grid.width - 1 - self.landmark_pos[0], self.grid.height - 1 - self.landmark_pos[1])
                elif self.task_mode == "target3":
                    invalid = abs(pos[0] - self.landmark_pos[0]) + abs(pos[1] - self.landmark_pos[1]) == 2
                elif self.task_mode == "target4":
                    invalid = pos == (self.grid.width - 1 - self.landmark_pos[0], self.grid.height - 1 - (self.landmark_pos[1] - 2))
                elif self.task_mode == "target5":
                    invalid = pos in (tuple(self.target_a_pos), tuple(self.target_b_pos))
                
                if invalid:
                    self.agent_pos = None
                    self.place_agent()
                else:
                    break

        self.mission = self.gen_mission()

    def get_description(self, obs):
        dir_names = {0: "right", 1: "down", 2: "left", 3: "up"}

        parts = [f"Agent is at ({self.agent_pos[0]}, {self.agent_pos[1]}) facing {dir_names[self.agent_dir]}."]

        if self.task_mode == "target5":
            # Each entity gets its own sentence (source-style: one
            # entity, one position per sentence). Completed pairs
            # disappear from the description so the model can read
            # the current phase off the input.
            #
            # Ball A
            if self.carrying is self.ball_a:
                parts.append(f"Agent is carrying the {self.ball_a.color} {self.ball_a.type}.")
            elif not self._target5_done["a"]:
                pos = getattr(self.ball_a, "cur_pos", None)
                if pos is not None:
                    parts.append(f"The {self.ball_a.color} {self.ball_a.type} is at ({pos[0]}, {pos[1]}).")
            # Ball B
            if self.carrying is self.ball_b:
                parts.append(f"Agent is carrying the {self.ball_b.color} {self.ball_b.type}.")
            elif not self._target5_done["b"]:
                pos = getattr(self.ball_b, "cur_pos", None)
                if pos is not None:
                    parts.append(f"The {self.ball_b.color} {self.ball_b.type} is at ({pos[0]}, {pos[1]}).")
            # Targets — only unfinished ones.
            if not self._target5_done["a"]:
                parts.append(f"The {self.target_a.color} target is at ({self.target_a_pos[0]}, {self.target_a_pos[1]}).")
            if not self._target5_done["b"]:
                parts.append(f"The {self.target_b.color} target is at ({self.target_b_pos[0]}, {self.target_b_pos[1]}).")
            return " ".join(parts)

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
        
        # Override Drop Action Logic to forcefully intercept on any
        # colored Floor target (green in source mode, yellow/grey in
        # target5). Target1-4 have no Floor tiles so this is a no-op
        # there.
        if action == self.actions.drop and self.carrying:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is not None and isinstance(fwd_cell, Floor):
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying = None

        self.previous_state = self.obs
        obs, dict_reward, terminated, truncated, info = super().step(action)
        
        # Base penalty for time
        reward = 0.0

        # add intermediate reward for picking up the correct tool ball
        if action == self.actions.pickup and carrying_before is None and self.carrying is not None:
            if self.task_mode == "target5":
                # Either ball is fair game in either order — but only
                # if its pair hasn't already been completed.
                if (self.carrying is self.ball_a and not self._target5_done["a"]) \
                        or (self.carrying is self.ball_b and not self._target5_done["b"]):
                    reward += 1.0
                    self.agent_performance["successful_pick"] += 1
                    if self.verbose:
                        print(f"Intermediate Success: Picked up the {self.carrying.color} ball!")
            else:
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
                # Compute the 4 cardinal surrounding cells of the landmark
                lx, ly = self.landmark_pos
                neighbors = [
                    (lx + dx, ly + dy)
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
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
                        print(f"Success! Dropped adjacent to yellow box at {tuple(drop_pos)}.")
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

            elif self.task_mode == "target3":
                # Cells at exactly Manhattan distance 2 from the yellow box,
                # then take the subset whose Manhattan distance to the
                # current agent position is minimal.
                lx, ly = self.landmark_pos
                ax, ay = self.agent_pos
                from minigrid.core.world_object import Wall

                ring = [
                    (lx + dx, ly + dy)
                    for dx in range(-2, 3)
                    for dy in range(-2, 3)
                    if abs(dx) + abs(dy) == 2
                ]
                # Stay inside the playable interior and skip walls.
                valid = [
                    pos for pos in ring
                    if 0 <= pos[0] < self.grid.width
                    and 0 <= pos[1] < self.grid.height
                    and not isinstance(self.grid.get(*pos), Wall)
                ]

                if not valid:
                    terminated = True
                    if self.verbose:
                        print(
                            f"Failed. No valid Manhattan-2 cells around "
                            f"yellow box at ({lx}, {ly})."
                        )
                else:
                    min_dist = min(
                        abs(px - ax) + abs(py - ay) for (px, py) in valid
                    )
                    nearest = [
                        pos for pos in valid
                        if abs(pos[0] - ax) + abs(pos[1] - ay) == min_dist
                    ]
                    if tuple(drop_pos) in nearest:
                        reward += 10.0
                        terminated = True
                        self.agent_performance["successful_drop"] += 1
                        if self.verbose:
                            print(
                                f"Success! Dropped at {tuple(drop_pos)} — "
                                f"Manhattan-2 from yellow box ({lx}, {ly}) "
                                f"and nearest to agent ({ax}, {ay})."
                            )
                    else:
                        terminated = True
                        if self.verbose:
                            print(
                                f"Failed. Dropped at {tuple(drop_pos)}; "
                                f"valid nearest cells were {nearest} "
                                f"(box at ({lx}, {ly}), agent at ({ax}, {ay}))."
                            )

            elif self.task_mode == "target4":
                lx, ly = self.landmark_pos
                tgt_x = (self.grid.width - 1) - lx
                tgt_y = (self.grid.height - 1) - (ly - 2)
                if tuple(drop_pos) == (tgt_x, tgt_y):
                    reward += 10.0
                    terminated = True
                    self.agent_performance["successful_drop"] += 1
                    if self.verbose:
                        print(
                            f"Success! Dropped at ({tgt_x}, {tgt_y}) — "
                            f"symmetric opposite of (box + 2*north) for "
                            f"yellow box at ({lx}, {ly})."
                        )
                else:
                    terminated = True
                    if self.verbose:
                        print(
                            f"Failed. Dropped at {tuple(drop_pos)}; "
                            f"expected ({tgt_x}, {tgt_y}) (box at ({lx}, {ly}))."
                        )

            elif self.task_mode == "target5":
                # Which ball was just dropped, and where should it
                # have gone?
                if carrying_before is self.ball_a:
                    which, want_pos = "a", tuple(self.target_a_pos)
                elif carrying_before is self.ball_b:
                    which, want_pos = "b", tuple(self.target_b_pos)
                else:
                    which, want_pos = None, None

                if which is None:
                    terminated = True
                    if self.verbose:
                        print("Failed. Dropped an unrecognised carry — terminating.")
                elif tuple(drop_pos) == want_pos:
                    reward += 10.0
                    self._target5_done[which] = True
                    self.agent_performance["successful_drop"] += 1
                    # Clear the cell: both the ball (which replaced
                    # the floor tile in the intercept above) and the
                    # tile itself disappear. This (1) lets the agent
                    # walk through that cell on the way to the other
                    # pair, and (2) prevents the model from being
                    # tempted to re-pick the placed ball.
                    self.grid.set(want_pos[0], want_pos[1], None)
                    carrying_before.cur_pos = None
                    if self.verbose:
                        print(
                            f"Success! Dropped the {carrying_before.color} ball "
                            f"on its matching target at {want_pos}; cell cleared."
                        )
                    # Episode ends only when BOTH pairs are done.
                    if self._target5_done["a"] and self._target5_done["b"]:
                        terminated = True
                        if self.verbose:
                            print("Success! Both pairs completed.")
                else:
                    terminated = True
                    if self.verbose:
                        print(
                            f"Failed. Dropped the {carrying_before.color} ball at "
                            f"{tuple(drop_pos)}; expected {want_pos}."
                        )

        self.obs = obs
        info["description"] = self.get_description(obs)

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        obs, info = super().reset(seed=seed, options=options)
        self.obs = obs
        if self.verbose:
            info["description"] = self.get_description(obs)
        return obs, info


# =====================================================================
# Visual rollout for manual inspection
# =====================================================================
#
# Run this file directly to sanity-check the env. It rolls a few
# episodes of a chosen `task_mode` with random actions, saves a
# rendered PNG per step under `out/inspect_env/<mode>/ep_NN/`, and
# writes a `log.txt` with the textual description, action, reward,
# and termination flag at every step.
#
# Example:
#     python env/minigrid_relational.py --mode target5 \
#         --episodes 3 --max-steps 30
#
# Random actions almost never solve the harder targets — that's fine.
# What you're checking is that the env constructs correctly, that
# `get_description()` reads out sensible sentences across phases, that
# the reward logic fires at the right moments, and (for target5) that
# both balls and both target tiles render correctly.
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from omegaconf import OmegaConf

    ACTION_NAMES = {
        0: "left", 1: "right", 2: "forward",
        3: "pickup", 4: "drop", 5: "toggle", 6: "done",
    }

    parser = argparse.ArgumentParser(
        description="Visual rollout of RelationalPickPlaceEnv for inspection.",
    )
    parser.add_argument(
        "--mode", default="target5",
        choices=["source", "target1", "target2", "target3", "target4", "target5"],
        help="task_mode to run (default: target5).",
    )
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes to roll out (default: 2).")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Per-episode step cap (default: 25).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; episode i uses seed + i.")
    parser.add_argument("--out-dir", default="out/inspect_env",
                        help="Where artefacts land (default: out/inspect_env).")
    parser.add_argument("--tile-size", type=int, default=16,
                        help="Render tile size in pixels (default: 16).")
    args = parser.parse_args()

    cfg = OmegaConf.create({
        "size": 8,
        "max_steps": args.max_steps,
        "verbose": True,
        "task_mode": args.mode,
    })

    base = RelationalPickPlaceEnv(cfg)
    env = ImgObsWrapper(RGBImgObsWrapper(base, tile_size=args.tile_size))

    out_root = Path(args.out_dir) / args.mode
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {args.mode} — saving artefacts to {out_root} ===\n")

    for ep in range(args.episodes):
        ep_dir = out_root / f"ep_{ep:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        obs, _ = env.reset(seed=args.seed + ep)
        Image.fromarray(obs).save(ep_dir / "step_00_init.png")

        log_path = ep_dir / "log.txt"
        with open(log_path, "w") as logf:
            logf.write(f"Mission : {base.mission}\n")
            logf.write(f"Mode    : {args.mode}    seed: {args.seed + ep}\n\n")
            logf.write("Step 00 (init)\n")
            logf.write(f"  description: {base.get_description(obs)}\n\n")

            done = False
            step = 0
            total_return = 0.0
            while not done and step < args.max_steps:
                step += 1
                action = env.action_space.sample()
                obs, r, term, trunc, info = env.step(action)
                total_return += float(r)
                Image.fromarray(obs).save(
                    ep_dir / f"step_{step:02d}_a{int(action)}.png"
                )
                logf.write(
                    f"Step {step:02d}  action={int(action)} ({ACTION_NAMES.get(int(action), '?')})  "
                    f"reward={float(r):+.1f}  term={bool(term)}  trunc={bool(trunc)}\n"
                    f"  description: {base.get_description(obs)}\n\n"
                )
                done = bool(term or trunc)

            logf.write("--- Episode complete ---\n")
            logf.write(f"Total return : {total_return:.1f}\n")
            logf.write(f"Steps taken  : {step}\n")
            logf.write(f"Performance  : {base.get_performance_metric()}\n")

        print(
            f"ep {ep:02d}  mode={args.mode}  steps={step}  "
            f"return={total_return:.2f}  perf={base.get_performance_metric()}"
        )

    env.close()
    print(f"\nDone. Open {out_root}/ to inspect the per-step PNGs and log.txt.")
