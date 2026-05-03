"""
PandaGymSimpleEnv: ASPECT-compatible wrapper around panda-gym's built-in
PandaPickAndPlaceDense-v3 / PandaPickAndPlace-v3.

Differences from PandaGymEnv.py:
  - Uses the registered gymnasium env directly (gym.make) — no custom cube
    resizing, no custom colours or visual markers.
  - All object sizes and workspace dimensions are panda-gym defaults.
  - Feature-based observations only (13D: EE pos/vel/fingers, cube pos, goal).
  - Same multi-phase reward and ASPECT interface as PandaGymEnv.py.

Source task : pick cube → place ON the goal (dist < near_threshold).
Target task : pick cube → place BESIDE a fixed landmark position
              (landmark_radius < dist < near_threshold * 2).
"""

import sys
import inspect
import numpy as np
import gymnasium as gym
from PIL import Image
from gymnasium import spaces
from omegaconf import DictConfig
import hydra

# ---- PyBullet patching (same as PandaGymEnv.py) -------------------------
import pybullet
import pybullet_utils.bullet_client as _bc

# _orig_sig = inspect.signature(_bc.BulletClient.__init__)
# if "options" not in _orig_sig.parameters:
#     _OrigBulletClient = _bc.BulletClient

#     class _PatchedBulletClient(_OrigBulletClient):
#         """BulletClient that silently accepts (and ignores) `options`."""
#         def __init__(self, connection_mode=None, options="", **kwargs):
#             if connection_mode is None:
#                 connection_mode = pybullet.DIRECT
#             self._client = pybullet.connect(connection_mode)
#             self._shapes = {}

#         def __getattr__(self, name):
#             attribute = getattr(pybullet, name)
#             if callable(attribute):
#                 import functools
#                 @functools.wraps(attribute)
#                 def wrapper(*args, **kw):
#                     return attribute(*args, physicsClientId=self._client, **kw)
#                 return wrapper
#             return attribute

#         def __del__(self):
#             if hasattr(self, '_client') and self._client >= 0:
#                 try:
#                     pybullet.disconnect(self._client)
#                 except Exception:
#                     pass

    # _bc.BulletClient = _PatchedBulletClient

import panda_gym  # noqa: registers gym envs

sys.path.append(".")

# ---- Default panda-gym dimensions (do NOT change — keep env as-is) ------
# panda-gym cube: object_size = 0.04  → half_extent = 0.02
# table surface z ≈ 0.0,  object rests at z = object_size/2 = 0.02
DEFAULT_CUBE_SIZE   = 0.04          # panda-gym default
DEFAULT_CUBE_HALF   = DEFAULT_CUBE_SIZE / 2.0   # 0.02
DEFAULT_GRASP_DIST  = DEFAULT_CUBE_SIZE * 1.5   # ~0.06 — generous heuristic

# Minimum XY separation between cube spawn and goal at reset
# sum of cube radii * 2 + small buffer to avoid overlap
MIN_SEPARATION = 2 * DEFAULT_CUBE_HALF + 0.06   # ~0.10 m

# Target task: "beside" landmark (a fixed position in space)
# near_threshold * 2 is the outer ring; landmark radius sets the inner ring
LANDMARK_RADIUS = 0.04   # inner exclusion radius (metres)


class PandaGymPickPlaceEnv(gym.Env):
    """
    Lightweight ASPECT-compatible wrapper for panda-gym pick-and-place.

    Uses the built-in PandaPickAndPlaceDense-v3 environment (all default
    dimensions / physics) and adds:
      - Multi-phase shaped reward
      - ASPECT text descriptions & sensor data
      - Performance metrics
      - Source / target task modes

    task_mode:
        'source' → place the cube ON the goal (dist_cube_goal < near_threshold)
        'target' → place the cube BESIDE a fixed landmark position
                   (LANDMARK_RADIUS < dist < near_threshold * 2)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()

        self.task_mode      = config.get("task_mode", "source")
        self.verbose        = config.get("verbose", False)
        self.max_steps      = config.get("max_steps", 100)
        self.reward_type    = config.get("reward_type", "dense")
        self.control_type   = config.get("control_type", "ee")
        self.near_threshold = config.get("near_threshold", 0.05)  # metres
        self.use_her        = config.get("use_her", False)

        # Build the gym id from reward_type
        # dense → PandaPickAndPlaceDense-v3
        # sparse → PandaPickAndPlace-v3
        if self.reward_type == "dense":
            gym_id = "PandaPickAndPlaceDense-v3"
        else:
            gym_id = "PandaPickAndPlace-v3"

        self._inner = gym.make(gym_id, render_mode="rgb_array",
                               control_type=self.control_type)

        # ---- Observation space ------------------------------------------
        # HER mode: Dict obs with separate achieved/desired goal keys
        #   observation  (25D): [EE pos | EE vel | fingers | cube pos | cube rot | cube lin vel | cube ang vel | target1 pos | target2 pos]
        #   achieved_goal (3D): current cube position
        #   desired_goal  (3D): target position based on task mode
        #
        # Flat mode (default): 25D vector
        #   [raw obs (19D) | target1 pos (3D) | target2 pos (3D)]
        self.obs_dim = 25
        if self.use_her:
            self.observation_space = spaces.Dict({
                "observation":   spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,),  dtype=np.float32),
                "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(3,),  dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
        self.action_space = self._inner.action_space

        # ---- ASPECT interface -------------------------------------------
        self.env_name = self._compute_env_name()
        self.env_description = self._get_environment_description()
        self.mission = self._gen_mission()

        # ---- Internal state --------------------------------------------
        self._step_count        = 0
        self._grasp_bonus_given = False
        self._target1_pos       = None   # blue landmark
        self._target2_pos       = None   # red landmark
        self._goal              = None   # pybullet visual goal
        self.obs                = None
        self.reset_metrices()

    # ------------------------------------------------------------------
    # ASPECT required properties / methods
    # ------------------------------------------------------------------
    def _compute_env_name(self):
        if self.task_mode == "source":
            return "PandaSimplePickPlaceOnTarget"
        elif self.task_mode == "target1":
            return "PandaSimplePickPlaceBesideTarget1"
        elif self.task_mode == "target2":
            return "PandaSimplePickPlaceBesideTarget2"
        else:
            return "PandaSimplePickPlaceMidpoint"

    def _gen_mission(self):
        if self.task_mode == "source":
            return "Pick up the cube and place it on the blue landmark."
        elif self.task_mode == "target1":
            return "Pick up the cube and place it beside the blue landmark."
        elif self.task_mode == "target2":
            return "Pick up the cube and place it beside the red landmark."
        else:
            return "Pick up the cube and place it at the midpoint between the blue and red landmarks."

    def _get_environment_description(self):
        return (
            "Environment context:\n"
            "- The agent is a Franka Emika Panda 7-DOF robotic arm with a parallel-jaw gripper.\n"
            "- The agent operates on a tabletop in a 3D PyBullet simulation.\n"
            "- The action space is 4D: 3D end-effector displacement (dx, dy, dz) and 1D gripper opening.\n"
            "- The workspace is roughly a 0.3 m x 0.3 m area centred at the origin on a table at z ≈ 0.02.\n"
            "- The cube has default panda-gym dimensions (side length 0.04 m).\n"
            "- The environment contains a blue landmark and a red landmark.\n"
            "- In the source task, the goal is the blue landmark.\n"
            "- In the target tasks, the goal is either beside the blue landmark, beside the red landmark, or at their midpoint.\n"
            "- Each episode ends when the task is completed or the maximum step limit is reached.\n"
        )

    def reset_metrices(self):
        self.agent_performance = {
            "successful_pick":  0,
            "successful_place": 0,
            "episodes":         0,
        }

    def get_performance_metric(self):
        return self.agent_performance

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs_dict, _ = self._inner.reset(seed=seed, options=options)

        self._step_count        = 0
        self._grasp_bonus_given = False

        # Extract initial goal from panda-gym as target1 (blue landmark)
        raw_goal = obs_dict["desired_goal"].copy()
        self._target1_pos = raw_goal

        # Enforce separation for both targets (this also samples target2)
        self._enforce_min_separation(obs_dict)

        # PyBullet visual target representation.
        if self.task_mode in ["source", "target1"]:
            self._set_goal(self._target1_pos)
        elif self.task_mode == "target2":
            self._set_goal(self._target2_pos)
        else:
            self._set_goal((self._target1_pos + self._target2_pos) / 2.0)

        observation = self._build_observation(obs_dict)

        info_out = {
            "description":    self.get_description(obs_dict),
            "sensor_data":    self.get_sensor_data(obs_dict),
            "is_success":     False,
        }

        self.agent_performance["episodes"] += 1
        return observation, info_out

    def _set_goal(self, goal: np.ndarray):
        """
        Store goal in self._goal and sync it into the panda-gym task and
        pybullet physics so the ghost target marker matches.
        """
        self._goal = goal.astype(np.float32)
        inner_task = getattr(self._inner.unwrapped, "task", None)
        if inner_task is not None:
            inner_task.goal = self._goal.copy()
        # Move the pybullet ghost target to match (no-op if name not found)
        try:
            sim = self._inner.unwrapped.sim
            sim.set_base_pose("target", self._goal, np.array([0.0, 0.0, 0.0, 1.0]))
        except Exception:
            pass

    def _enforce_min_separation(self, obs_dict):
        """
        Resample goals to ensure cube, target1, and target2 are well separated.
        """
        cube_pos = obs_dict["achieved_goal"].copy()
        inner_task = getattr(self._inner.unwrapped, "task", None)

        # 1) Fix target 1 separation from cube
        for _ in range(50):
            xy_dist = np.linalg.norm(cube_pos[:2] - self._target1_pos[:2])
            if xy_dist >= MIN_SEPARATION:
                break
            
            if inner_task is not None and hasattr(inner_task, "_sample_goal"):
                new_goal = inner_task._sample_goal()
            else:
                new_goal = np.array([
                    np.random.uniform(-0.15, 0.15),
                    np.random.uniform(-0.15, 0.15),
                    np.random.uniform(-0.15, 0.15),
                ])
            self._target1_pos = new_goal

        # 2) Sample target 2, ensuring separation from both cube AND target 1
        self._target2_pos = self._target1_pos.copy()
        for _ in range(50):
            new_t2 = np.array([
                np.random.uniform(-0.15, 0.15),
                np.random.uniform(-0.15, 0.15),
                np.random.uniform(-0.15, 0.15),
            ])
            dist_c = np.linalg.norm(cube_pos[:2] - new_t2[:2])
            dist_1 = np.linalg.norm(self._target1_pos[:2] - new_t2[:2])
            # Well separated (at least 1.5 * MIN_SEPARATION between targets)
            if dist_c >= MIN_SEPARATION and dist_1 >= MIN_SEPARATION * 1.5:
                self._target2_pos = new_t2
                break

        if self.verbose:
            dist_c1 = np.linalg.norm(cube_pos[:2] - self._target1_pos[:2])
            dist_c2 = np.linalg.norm(cube_pos[:2] - self._target2_pos[:2])
            dist_12 = np.linalg.norm(self._target1_pos[:2] - self._target2_pos[:2])
            if dist_c1 < MIN_SEPARATION or dist_c2 < MIN_SEPARATION or dist_12 < MIN_SEPARATION * 1.5:
                print("[WARN] _enforce_min_separation: could not find fully separated goals")

    def step(self, action):
        obs_dict, _inner_reward, terminated, truncated, _inner_info = self._inner.step(action)
        self._step_count += 1

        # ---- Extract positions ------------------------------------------
        cube_pos = obs_dict["achieved_goal"]          # current cube position
        ee_pos   = obs_dict["observation"][:3]         # end-effector position

        reward       = 0.0
        is_grasping  = self._is_grasping(obs_dict)

        # ---- One-time grasp bonus ----------------------------------------
        if is_grasping and not self._grasp_bonus_given:
            self._grasp_bonus_given = True
            reward += 5.0
            self.agent_performance["successful_pick"] += 1
            if self.verbose:
                print("Intermediate: Grasped the cube!")

        # ---- Goal position -----------------------------------------------
        if self.task_mode == "source" or self.task_mode == "target1":
            goal_pos = self._target1_pos
        elif self.task_mode == "target2":
            goal_pos = self._target2_pos
        else: # target3
            goal_pos = (self._target1_pos + self._target2_pos) / 2.0

        dist_ee_cube   = np.linalg.norm(ee_pos - cube_pos)
        dist_cube_goal = np.linalg.norm(cube_pos - goal_pos)

        # ---- Shaped reward (hack-proof) ----------------------------------
        # Always penalise cube-to-goal distance so dropping never helps.
        # Additionally penalise EE-to-cube distance when not grasping.
        if self.task_mode == "source" or self.task_mode == "target3":
            reward += -dist_cube_goal
        else:
            ideal_dist = (LANDMARK_RADIUS + self.near_threshold * 2) / 2.0
            reward += -abs(dist_cube_goal - ideal_dist)

        if not is_grasping:
            reward += -dist_ee_cube

        # ---- Placement success ------------------------------------------
        if self.task_mode == "source" or self.task_mode == "target3":
            if dist_cube_goal < self.near_threshold:
                reward += 10.0
                terminated = True
                self.agent_performance["successful_place"] += 1
                if self.verbose:
                    print(f"Success! Placed on point (dist={dist_cube_goal:.3f}).")
        else:
            if LANDMARK_RADIUS < dist_cube_goal <= self.near_threshold * 2:
                reward += 10.0
                terminated = True
                self.agent_performance["successful_place"] += 1
                if self.verbose:
                    print(f"Success! Placed beside landmark (dist={dist_cube_goal:.3f}).")

        if self._step_count >= self.max_steps:
            truncated = True

        observation = self._build_observation(obs_dict)

        info_out = {
            "description": self.get_description(obs_dict),
            "sensor_data":  self.get_sensor_data(obs_dict),
            # HER requires is_success in info
            "is_success":   terminated and self.agent_performance["successful_place"] > 0,
        }

        return observation, float(reward), terminated, truncated, info_out

    def _build_observation(self, obs_dict):
        """
        Build the observation returned to the agent.

        Raw panda-gym observation layout (19D):
            obs[0:3]  → EE position
            obs[3:6]  → EE velocity
            obs[6]    → fingers width
            obs[7:10] → object (cube) position
            obs[10:13]→ object rotation (Euler)
            obs[13:16]→ object linear velocity
            obs[16:19]→ object angular velocity

        HER mode  → Dict{observation(25D), achieved_goal(3D), desired_goal(3D)}
        Flat mode → 25D concatenation [raw (19D) | target1 pos (3D) | target2 pos (3D)]
        """
        raw      = obs_dict["observation"].astype(np.float32)
        cube_pos = raw[7:10]
        
        if self.task_mode == "source" or self.task_mode == "target1":
            goal_pos = self._target1_pos
        elif self.task_mode == "target2":
            goal_pos = self._target2_pos
        else:
            goal_pos = (self._target1_pos + self._target2_pos) / 2.0

        target1 = self._target1_pos if self._target1_pos is not None else np.zeros(3)
        target2 = self._target2_pos if self._target2_pos is not None else np.zeros(3)
        full_obs = np.concatenate([raw, target1, target2]).astype(np.float32)

        if self.use_her:
            # Dict observation required by HerReplayBuffer
            self.obs = {
                "observation":   full_obs,              # 25D
                "achieved_goal": cube_pos.copy(),       # 3D — current cube pos
                "desired_goal":  goal_pos.copy(),       # 3D — target pos
            }
        else:
            self.obs = full_obs
        return self.obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Required by HerReplayBuffer for goal relabelling.

        Called with batched arrays: achieved_goal & desired_goal are
        (batch_size, 3) arrays of cube positions and goal positions.

        Returns dense reward.
        For sparse reward_type: 0 if within threshold, -1 otherwise.
        """
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        
        if self.task_mode in ["target1", "target2"]:
            # Beside target logic
            ideal_dist = (LANDMARK_RADIUS + self.near_threshold * 2) / 2.0
            dense_reward = -np.abs(dist - ideal_dist)
            sparse_reward = -( (dist <= LANDMARK_RADIUS) | (dist > self.near_threshold * 2) ).astype(np.float32)
        else:
            # On target logic (source or target3)
            dense_reward = -dist
            sparse_reward = -(dist > self.near_threshold).astype(np.float32)

        if self.reward_type == "sparse":
            return sparse_reward
        else:
            return dense_reward.astype(np.float32)

    # ------------------------------------------------------------------
    # Description & sensor data (ASPECT interface)
    # ------------------------------------------------------------------
    def get_description(self, obs_dict=None):
        """Structured text caption of the current scene."""
        if obs_dict is None:
            obs_dict = self._get_inner_obs()

        ee_pos   = obs_dict["observation"][:3]
        cube_pos = obs_dict["achieved_goal"]
        grasping = self._is_grasping(obs_dict)

        parts = [
            f"Gripper is at ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f})."
        ]
        if grasping:
            parts.append("The gripper is grasping the cube.")
        else:
            parts.append(
                f"The cube is at ({cube_pos[0]:.2f}, {cube_pos[1]:.2f}, {cube_pos[2]:.2f})."
            )

        if self._target1_pos is not None and self._target2_pos is not None:
            t1 = self._target1_pos
            t2 = self._target2_pos
            parts.append(f"The blue landmark is at ({t1[0]:.2f}, {t1[1]:.2f}, {t1[2]:.2f}).")
            parts.append(f"The red landmark is at ({t2[0]:.2f}, {t2[1]:.2f}, {t2[2]:.2f}).")

        if self.task_mode == "source":
            parts.append("The cube must be placed on the blue landmark.")
        elif self.task_mode == "target1":
            parts.append("The cube must be placed beside the blue landmark.")
        elif self.task_mode == "target2":
            parts.append("The cube must be placed beside the red landmark.")
        elif self.task_mode == "target3":
            parts.append("The cube must be placed at the midpoint between the blue and red landmarks.")

        return " ".join(parts)

    def get_sensor_data(self, obs_dict=None):
        """Raw numeric sensor data string."""
        if obs_dict is None:
            obs_dict = self._get_inner_obs()

        ee_pos   = obs_dict["observation"][:3]
        cube_pos = obs_dict["achieved_goal"]

        parts = [
            f"ee: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
            f"cube: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]",
        ]
        if self._target1_pos is not None:
            t1 = self._target1_pos
            parts.append(f"blue_landmark: [{t1[0]:.3f}, {t1[1]:.3f}, {t1[2]:.3f}]")
        if self._target2_pos is not None:
            t2 = self._target2_pos
            parts.append(f"red_landmark: [{t2[0]:.3f}, {t2[1]:.3f}, {t2[2]:.3f}]")

        return "Sensor data: " + "; ".join(parts)

    def get_frame(self):
        """Return the current rendered frame (for video recording)."""
        frame = self._inner.render()
        if frame is None:
            return np.zeros((84, 84, 3), dtype=np.uint8)
        return frame.astype(np.uint8)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_inner_obs(self):
        """Re-fetch the current observation dict from the inner env."""
        try:
            inner = self._inner.unwrapped
            robot_obs   = inner.robot.get_obs()
            task_obs    = inner.task.get_obs()
            achieved    = inner.task.get_achieved_goal()
            desired     = inner.task.get_goal()
            return {
                "observation":   np.concatenate([robot_obs, task_obs]),
                "achieved_goal": achieved,
                "desired_goal":  desired,
            }
        except Exception:
            return {
                "observation":   np.zeros(19, dtype=np.float32),
                "achieved_goal": np.zeros(3,  dtype=np.float32),
                "desired_goal":  self._target1_pos.copy() if self._target1_pos is not None else np.zeros(3),
            }

    def _is_grasping(self, obs_dict):
        """
        Grasp detection: gripper fingers are partially closed around the cube.

        Uses finger width as the primary signal (more reliable than elevation):
          - fingers_width < GRASP_FINGER_THRESHOLD  → gripper is closing/closed
          - dist_ee_cube < DEFAULT_GRASP_DIST        → EE is near the cube
          - cube_pos[2] > TABLE_Z                    → cube has left the table surface

        Thresholds derived from panda-gym defaults:
          - Fully open finger width ≈ 0.08 m
          - Cube width = 0.04 m → fingers must be < 0.05 m to grip it
          - Table surface z ≈ 0.0; cube rests at z = 0.02; lifted means z > 0.025
        """
        raw      = obs_dict["observation"]
        ee_pos   = raw[0:3]
        fingers  = float(raw[6])          # total finger gap in metres (~0.0–0.08)
        cube_pos = obs_dict["achieved_goal"]

        dist_ee_cube = np.linalg.norm(ee_pos - cube_pos)

        fingers_closed = fingers < 0.05                    # gripper squeezing cube
        ee_near_cube   = dist_ee_cube < DEFAULT_GRASP_DIST # EE close enough
        cube_lifted    = cube_pos[2] > DEFAULT_CUBE_HALF   # above resting height

        return fingers_closed and ee_near_cube and cube_lifted

    def close(self):
        try:
            self._inner.close()
        except Exception:
            pass

    @property
    def unwrapped(self):
        return self


# ======================================================================
# Standalone testing
# ======================================================================
@hydra.main(version_base=None, config_path="../config/env", config_name="PandaGymSimple")
def main(args: DictConfig) -> None:
    env = PandaGymSimpleEnv(args)
    print("Mission:", env.mission)
    print("Env name:", env.env_name)
    print("Obs space:", env.observation_space)
    print("Act space:", env.action_space)

    for episode in range(5):
        obs, info = env.reset()
        print(f"\n--- Episode {episode + 1} ---")
        print("Obs shape:", obs.shape, "| dtype:", obs.dtype)
        print("Description:", info["description"])

        done = False
        steps = 0
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc
            steps += 1

        print(f"Ended after {steps} steps | total reward: {total_reward:.3f}")

    print("\nPerformance:", env.get_performance_metric())
    env.close()


if __name__ == "__main__":
    main()
