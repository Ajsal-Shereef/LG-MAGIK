import os
import math
import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw
from gymnasium import spaces
from omegaconf import DictConfig

class CollisionEnv(gym.Env):
    """
    2D Continuous Elastic Collision Environment.
    The agent must apply 2D drive forces to hit/nudge an object (light/heavy circle or square)
    into a designated target destination zone using elastic collision momentum transfer,
    without exceeding the shattering impact force threshold.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: DictConfig, render_mode="rgb_array"):
        super().__init__()

        self.name = cfg.get("name", "CollisionEnv")
        self.width = int(cfg.get("width", 128))
        self.height = int(cfg.get("height", 128))
        self.max_steps = int(cfg.get("max_steps", 100))
        self.render_mode = render_mode

        # Physics parameters
        self.agent_mass = float(cfg.get("agent_mass", 1.0))
        self.light_mass = float(cfg.get("light_mass", 0.5))
        self.heavy_mass = float(cfg.get("heavy_mass", 2.5))
        self.elasticity = float(cfg.get("elasticity", 0.85))  # Coefficient of restitution e in [0, 1]
        self.linear_drag = float(cfg.get("linear_drag", 0.15)) # Ground friction / drag coefficient
        self.max_drive_force = float(cfg.get("max_drive_force", 5.0))
        self.dt = float(cfg.get("dt", 0.1))

        # Shattering thresholds (max allowable impact force = J / dt)
        self.shatter_force_light = float(cfg.get("shatter_force_light", 4.0))
        self.shatter_force_heavy = float(cfg.get("shatter_force_heavy", 10.0))

        # Destination and reach
        self.goal_radius = float(cfg.get("goal_radius", 12.0))
        self.reach_margin = float(cfg.get("reach_margin", 3.0))

        # Reward coefficients
        self.shatter_penalty = float(cfg.get("shatter_penalty", 5.0))
        self.success_reward = float(cfg.get("success_reward", 10.0))
        self.step_penalty = float(cfg.get("step_penalty", 0.05))
        self.progress_reward_coeff = float(cfg.get("progress_reward_coeff", 1.0))

        # Visual quality
        self.ssaa_scale = int(cfg.get("ssaa_scale", 3))
        self.render_scale = int(cfg.get("render_scale", 2))
        self.agent_shape = cfg.get("agent_shape", "disk")

        self.task_mode = cfg.get("task_mode", "target")
        self.mode = cfg.get("mode", "train")
        self.verbose = bool(cfg.get("verbose", True))
        self.observation_mode = cfg.get("observation_mode", "feature")

        # 2D Action Space: Continuous force vector [force_x, force_y] in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )

        # 16-dimensional continuous feature representation
        if self.observation_mode == "feature":
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(16,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )

        self.screen = None
        self.clock = None
        self.last_action_status = None
        self.num_success = 0
        self.num_shattered = 0

        self.reset_state()
        self.mission = self._get_mission()
        self.action_dict = self.get_action_dict()

    def get_action_dict(self):
        """
        Returns structured dictionary mapping 2D control actions and physical capabilities.
        """
        return {
            "action_format": "Continuous 2D control vector [force_x, force_y] with values in [-1.0, 1.0].",
            "force_x": "Drive force along the horizontal X-axis in [-1.0, 1.0] (-1.0=full left thrust, +1.0=full right thrust).",
            "force_y": "Drive force along the vertical Y-axis in [-1.0, 1.0] (-1.0=full upward thrust, +1.0=full downward thrust).",
            "physics_rules": (
                "Elastic Collision: When the agent hits the object, momentum is transferred along the collision normal. "
                "Shattering Warning: If the collision impact force (J / dt) exceeds the object's fracture threshold "
                f"(Light: {self.shatter_force_light:.1f} N, Heavy: {self.shatter_force_heavy:.1f} N), the object shatters and the episode terminates with a penalty. "
                "Delicate control: Decelerate or apply gentle nudges to guide the object into the destination circle."
            )
        }

    def _get_mission(self):
        if self.task_mode == "source":
            return "Hit and guide the light circle or heavy square into the destination zone without shattering it."
        else:
            return "Hit and guide the heavy circle or light square into the destination zone without shattering it."

    def reset_state(self):
        margin = int(self.width * 0.15)
        self.agent_radius = float(self.width / 20.0)

        # Agent initial state: 2D Position and 2D Velocity
        self.agent_pos = np.array([
            self.np_random.uniform(margin, self.width - margin),
            self.np_random.uniform(margin, self.height - margin)
        ], dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)

        # Object selection based on task_mode
        if self.task_mode == "source":
            obj_type = self.np_random.choice(["circle", "square"])
            obj_weight = "light" if obj_type == "circle" else "heavy"
        else:
            obj_type = self.np_random.choice(["circle", "square"])
            obj_weight = "heavy" if obj_type == "circle" else "light"

        obj_mass = self.light_mass if obj_weight == "light" else self.heavy_mass
        shatter_thresh = self.shatter_force_light if obj_weight == "light" else self.shatter_force_heavy

        self.object = {
            "type": obj_type,
            "weight": obj_weight,
            "mass": obj_mass,
            "shatter_threshold": shatter_thresh,
            "radius": float(self.width / 18.0),
            "size": float(self.width / 14.0),
            "pos": np.zeros(2, dtype=np.float32),
            "vel": np.zeros(2, dtype=np.float32),
            "shattered": False,
            "in_goal": False
        }

        # Place object away from agent
        for _ in range(50):
            self.object["pos"] = np.array([
                self.np_random.uniform(margin, self.width - margin),
                self.np_random.uniform(margin, self.height - margin)
            ], dtype=np.float32)
            if np.linalg.norm(self.agent_pos - self.object["pos"]) > (self.agent_radius + self.object["radius"] + 15.0):
                break

        # Place Destination Goal away from object and agent
        for _ in range(50):
            self.goal_pos = np.array([
                self.np_random.uniform(margin, self.width - margin),
                self.np_random.uniform(margin, self.height - margin)
            ], dtype=np.float32)
            if (np.linalg.norm(self.goal_pos - self.object["pos"]) > 25.0 and
                np.linalg.norm(self.goal_pos - self.agent_pos) > 20.0):
                break

        self.steps = 0
        self.done = False
        self.last_action_status = None
        self.last_impact_force = 0.0

    def _get_description(self):
        if self.object["shattered"]:
            return "Object has shattered into pieces due to high impact collision! Episode terminated."
        if self.object["in_goal"]:
            return "Object has safely reached the destination goal zone! Episode succeeded."

        # Relative 2D vectors
        scale = self.width / 10.0
        rel_obj = (self.object["pos"] - self.agent_pos) / scale
        dist_obj = float(np.linalg.norm(rel_obj))

        rel_goal = (self.goal_pos - self.object["pos"]) / scale
        dist_goal = float(np.linalg.norm(rel_goal))

        weight = self.object["weight"]
        otype = self.object["type"]

        va_speed = float(np.linalg.norm(self.agent_vel))
        vo_speed = float(np.linalg.norm(self.object["vel"]))

        return (
            f"Agent sees a {weight} {otype} at relative position ({rel_obj[0]:.2f}, {rel_obj[1]:.2f}) ({dist_obj:.1f} units away). "
            f"Destination goal is ({rel_goal[0]:.2f}, {rel_goal[1]:.2f}) relative to object ({dist_goal:.1f} units away). "
            f"Agent speed: {va_speed:.2f} units/s (vel: [{self.agent_vel[0]:.2f}, {self.agent_vel[1]:.2f}]), "
            f"Object speed: {vo_speed:.2f} units/s (vel: [{self.object['vel'][0]:.2f}, {self.object['vel'][1]:.2f}])."
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_state()
        self.mission = self._get_mission()
        info = {
            "mission": self.mission,
            "description": self._get_description(),
            "target_task": self.mission
        }
        return self._get_obs(), info

    def step(self, action):
        self.steps += 1
        self.last_action_status = None
        self.last_impact_force = 0.0

        # Action command: 2D drive force in [-1, 1]
        drive_cmd = np.clip(np.array(action, dtype=np.float32)[:2], -1.0, 1.0)
        applied_force = drive_cmd * self.max_drive_force

        dist_to_goal_before = float(np.linalg.norm(self.object["pos"] - self.goal_pos))

        # 1. Update Agent Dynamics (Newton's 2nd Law: a = (F - drag * v) / m)
        agent_acc = (applied_force - self.linear_drag * self.agent_vel) / self.agent_mass
        self.agent_vel += agent_acc * self.dt
        self.agent_pos += self.agent_vel * self.dt * (self.width / 10.0)

        # 2. Update Object Dynamics (Ground Friction / Drag)
        obj_drag_acc = - (self.linear_drag * self.object["vel"]) / self.object["mass"]
        self.object["vel"] += obj_drag_acc * self.dt
        self.object["pos"] += self.object["vel"] * self.dt * (self.width / 10.0)

        # Arena boundary handling (elastic rebound from arena walls)
        self._handle_wall_collisions()

        # 3. Collision Detection between Agent and Object
        delta_pos = self.object["pos"] - self.agent_pos
        dist_centers = float(np.linalg.norm(delta_pos))
        min_dist = self.agent_radius + self.object["radius"]

        reward = -self.step_penalty
        terminated = False
        truncated = False

        if dist_centers < min_dist and dist_centers > 1e-6:
            # Collision normal pointing from agent to object
            normal = delta_pos / dist_centers
            # Relative approach velocity along normal
            rel_vel = self.agent_vel - self.object["vel"]
            approach_speed = float(np.dot(rel_vel, normal))

            if approach_speed > 0:  # Moving towards each other
                # 2D Elastic Collision Impulse J = (1 + e) * v_rel / (1/m1 + 1/m2)
                reduced_mass = 1.0 / (1.0 / self.agent_mass + 1.0 / self.object["mass"])
                impulse = (1.0 + self.elasticity) * approach_speed * reduced_mass
                impact_force = impulse / self.dt
                self.last_impact_force = impact_force

                # Check shattering threshold
                if impact_force >= self.object["shatter_threshold"]:
                    self.object["shattered"] = True
                    self.last_action_status = "shattered"
                    self.num_shattered += 1
                    reward -= self.shatter_penalty
                    terminated = True
                else:
                    # Valid elastic collision: Update post-collision velocities
                    self.agent_vel -= (impulse / self.agent_mass) * normal
                    self.object["vel"] += (impulse / self.object["mass"]) * normal
                    self.last_action_status = "hit"

                    # Push apart slightly to resolve overlap
                    overlap = min_dist - dist_centers
                    self.agent_pos -= normal * (overlap * 0.5)
                    self.object["pos"] += normal * (overlap * 0.5)

        # 4. Goal Progress & Success Check (only if object has not shattered)
        if not self.object["shattered"]:
            dist_to_goal_after = float(np.linalg.norm(self.object["pos"] - self.goal_pos))
            progress = (dist_to_goal_before - dist_to_goal_after) / (self.width / 10.0)
            reward += self.progress_reward_coeff * progress

            # Check Goal Success: Inside destination radius with settled speed
            if dist_to_goal_after <= self.goal_radius:
                obj_speed = float(np.linalg.norm(self.object["vel"]))
                if obj_speed < 1.5:
                    self.object["in_goal"] = True
                    self.last_action_status = "success"
                    self.num_success += 1
                    reward += self.success_reward
                    terminated = True
                else:
                    # Apply destination zone damping to assist in settling guided objects
                    self.object["vel"] *= 0.8

        if self.steps >= self.max_steps:
            truncated = True

        if self.verbose:
            info = {
                "mission": self.mission,
                "description": self._get_description(),
                "impact_force": self.last_impact_force,
                "status": self.last_action_status
            }
        else:
            info = {"mission": self.mission}

        return self._get_obs(), reward, terminated, truncated, info

    def _handle_wall_collisions(self):
        bounds = (self.width, self.height)
        # Agent bounce
        ar = self.agent_radius
        for i in (0, 1):
            if self.agent_pos[i] < ar:
                self.agent_pos[i] = ar
                self.agent_vel[i] = -self.agent_vel[i] * 0.5
            elif self.agent_pos[i] > bounds[i] - ar:
                self.agent_pos[i] = bounds[i] - ar
                self.agent_vel[i] = -self.agent_vel[i] * 0.5

        # Object bounce
        obr = self.object["radius"]
        for i in (0, 1):
            if self.object["pos"][i] < obr:
                self.object["pos"][i] = obr
                self.object["vel"][i] = -self.object["vel"][i] * 0.5
            elif self.object["pos"][i] > bounds[i] - obr:
                self.object["pos"][i] = bounds[i] - obr
                self.object["vel"][i] = -self.object["vel"][i] * 0.5

    def _get_obs(self):
        if self.observation_mode == "feature":
            return self._get_features()
        return self.get_frame()

    def _get_features(self):
        """16-dimensional continuous feature representation"""
        # Normalized coordinates and velocities
        bounds = np.array([self.width, self.height], dtype=np.float32)
        ap_norm = np.clip((self.agent_pos / bounds) * 2.0 - 1.0, -1.0, 1.0)
        av_norm = np.clip(self.agent_vel / 5.0, -1.0, 1.0)

        op_norm = np.clip((self.object["pos"] / bounds) * 2.0 - 1.0, -1.0, 1.0)
        ov_norm = np.clip(self.object["vel"] / 5.0, -1.0, 1.0)

        gp_norm = np.clip((self.goal_pos / bounds) * 2.0 - 1.0, -1.0, 1.0)
        rel_goal_norm = np.clip((self.goal_pos - self.object["pos"]) / bounds, -1.0, 1.0)

        is_circle = 1.0 if self.object["type"] == "circle" else 0.0
        is_light = 1.0 if self.object["weight"] == "light" else 0.0
        is_shattered = 1.0 if self.object["shattered"] else 0.0
        is_in_goal = 1.0 if self.object["in_goal"] else 0.0

        return np.array([
            ap_norm[0], ap_norm[1],
            av_norm[0], av_norm[1],
            op_norm[0], op_norm[1],
            ov_norm[0], ov_norm[1],
            gp_norm[0], gp_norm[1],
            rel_goal_norm[0], rel_goal_norm[1],
            is_circle, is_light,
            is_shattered, is_in_goal
        ], dtype=np.float32)

    def get_frame(self):
        """Renders high-quality RGB frame array for GIF logging and visual models."""
        W, H = self.width, self.height
        surface = pygame.Surface((W, H))
        surface.fill((30, 35, 45))  # Modern dark background

        # 1. Draw Destination Goal Zone (Glowing cyan/blue concentric ring)
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        gr = int(self.goal_radius)
        gfxdraw.filled_circle(surface, gx, gy, gr, (0, 180, 216, 80))
        gfxdraw.aacircle(surface, gx, gy, gr, (0, 230, 255))
        gfxdraw.filled_circle(surface, gx, gy, max(2, int(gr * 0.3)), (255, 255, 255))

        # 2. Draw Object (Green for light, Red for heavy)
        if not self.object["shattered"]:
            color = (46, 204, 113) if self.object["weight"] == "light" else (231, 76, 60)
            ox, oy = int(self.object["pos"][0]), int(self.object["pos"][1])

            if self.object["type"] == "circle":
                r = int(self.object["radius"])
                gfxdraw.filled_circle(surface, ox, oy, r, color)
                gfxdraw.aacircle(surface, ox, oy, r, (255, 255, 255))
            else:
                sz = int(self.object["size"])
                rect = pygame.Rect(ox - sz // 2, oy - sz // 2, sz, sz)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (255, 255, 255), rect, 1)

            # Draw Object Velocity Vector indicator
            if np.linalg.norm(self.object["vel"]) > 0.1:
                vx_end = int(ox + self.object["vel"][0] * 3.0)
                vy_end = int(oy + self.object["vel"][1] * 3.0)
                pygame.draw.line(surface, (255, 255, 0), (ox, oy), (vx_end, vy_end), 2)
        else:
            # Shattered debris representation
            ox, oy = int(self.object["pos"][0]), int(self.object["pos"][1])
            for offset in [(-6, -6), (6, -4), (-4, 6), (5, 5), (0, -7)]:
                gfxdraw.filled_circle(surface, ox + offset[0], oy + offset[1], 2, (180, 50, 50))

        # 3. Draw Agent (Vibrant Amber/Gold disk with direction indicator)
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        ar = int(self.agent_radius)
        agent_color = (241, 196, 15)  # Bright yellow/gold
        gfxdraw.filled_circle(surface, ax, ay, ar, agent_color)
        gfxdraw.aacircle(surface, ax, ay, ar, (255, 255, 255))

        # Draw Agent Velocity indicator
        if np.linalg.norm(self.agent_vel) > 0.1:
            ax_end = int(ax + self.agent_vel[0] * 3.0)
            ay_end = int(ay + self.agent_vel[1] * 3.0)
            pygame.draw.line(surface, (255, 255, 255), (ax, ay), (ax_end, ay_end), 2)

        # 4. Status outline
        if self.last_action_status == "shattered":
            pygame.draw.rect(surface, (255, 0, 0), pygame.Rect(0, 0, W, H), 3)
        elif self.last_action_status == "success":
            pygame.draw.rect(surface, (0, 255, 0), pygame.Rect(0, 0, W, H), 3)

        arr = np.transpose(np.array(pygame.surfarray.pixels3d(surface), copy=True), (1, 0, 2))
        return arr.astype(np.uint8)

    def render(self):
        if self.render_mode == "rgb_array":
            return self.get_frame()
        elif self.render_mode == "human":
            frame = self.get_frame()
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption(self.name)
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self.screen.blit(surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return None

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def get_performance_metric(self):
        return {
            "success": self.num_success,
            "shattered": self.num_shattered
        }

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/env/CollisionEnv.yaml")
    env = CollisionEnv(cfg)
    obs, info = env.reset(seed=42)
    print("Initial observation feature shape:", obs.shape)
    print("Mission:", info["mission"])
    print("Initial Caption:", info["description"])
    print("Action Dict:\n", env.get_action_dict())

    # Test taking a step
    obs, reward, term, trunc, info = env.step(np.array([0.5, -0.5], dtype=np.float32))
    print(f"\nStep 1 Result -> Reward: {reward:.4f}, Terminated: {term}, Impact Force: {info['impact_force']:.2f}")
    print("Frame shape:", env.get_frame().shape)
