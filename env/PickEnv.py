import os
import json
import hydra
import math
import gymnasium as gym
import numpy as np
import pygame
from PIL import Image
from pygame import gfxdraw
from gymnasium import spaces
from omegaconf import DictConfig
import sys
sys.path.append(".")
from architectures.common_utils import rollout, collect_data

class PickEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: DictConfig, mode="train", render_mode="rgb_array"):
        super().__init__()

        # Config (strict: accept only cfg)
        self.name = cfg.name 
        self.width = cfg.width
        self.height = cfg.height
        self.max_steps = cfg.max_steps
        self.light_weight_threshold = cfg.light_weight_threshold
        self.heavy_weight_threshold = cfg.heavy_weight_threshold
        self.approach_force_penalty = cfg.approach_force_penalty
        self.force_penalty_coeff = cfg.force_penalty_coeff
        self.approach_reward_coeff = cfg.approach_reward_coeff
        self.verbose = cfg.verbose
        self.render_mode = render_mode

        # Visual quality from cfg
        self.ssaa_scale = cfg.ssaa_scale
        self.render_scale = cfg.render_scale
        self.agent_shape = cfg.agent_shape

        # Mode: "data_collection" or "train"
        self.mode = mode

        # max per-step angle deviation at dist=1 (add to Hydra config)
        self.max_turn_rads = getattr(cfg, "max_turn_rads", 0.5)  # radians at dist=1
        
        self.observation_mode = getattr(cfg, "observation_mode", "images")

        # Actions: [angle_delta_cmd, distance, force], all in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )

        if self.observation_mode == "features":
            # 12 features: sin/cos angle, rel pos, rel bearing, one-hot type/weight, status
            feature_dim = 12
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(feature_dim,), dtype=np.float32
            )
        else: # Default to image
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8
            )

        # Internal state
        self.reset_state()

        # Pygame display state
        self.screen = None
        self.clock = None

    # ---------------------------
    # Core environment methods
    # ---------------------------
    def reset_state(self):
        # Agent - use dynamic margin (15% of dimension, with minimum of 10 pixels)
        agent_margin_x = max(10, int(self.width * 0.15))
        agent_margin_y = max(10, int(self.height * 0.15))
        
        # Ensure valid range for agent position
        agent_min_x = min(agent_margin_x, self.width // 2)
        agent_max_x = max(agent_min_x + 1, self.width - agent_margin_x)
        agent_min_y = min(agent_margin_y, self.height // 2)
        agent_max_y = max(agent_min_y + 1, self.height - agent_margin_y)
        
        self.agent_pos = np.array([
            self.np_random.integers(agent_min_x, agent_max_x),
            self.np_random.integers(agent_min_y, agent_max_y)
        ], dtype=np.float32)
        self.agent_angle = float(self.np_random.uniform(-np.pi, np.pi))
        self.agent_radius = self.width / 20.0  # scales with resolution
    
        # Object initialization based on mode
        if self.mode == "train":
            # Train mode: only light ball (circle) and heavy square
            obj_type = self.np_random.choice(["circle", "square"])
            if obj_type == "circle":
                obj_weight = "light"  # light ball
            else:
                obj_weight = "heavy"  # heavy square
        else:
            # data_collection mode: random type and weight
            obj_type = self.np_random.choice(["circle", "square"])
            obj_weight = self.np_random.choice(["light", "heavy"])
    
        # Dynamic margin for objects (20% of dimension, with minimum of 15 pixels)
        obj_margin_x = max(15, int(self.width * 0.20))
        obj_margin_y = max(15, int(self.height * 0.20))
        
        # Ensure valid range for object position
        obj_min_x = min(obj_margin_x, self.width // 2)
        obj_max_x = max(obj_min_x + 1, self.width - obj_margin_x)
        obj_min_y = min(obj_margin_y, self.height // 2)
        obj_max_y = max(obj_min_y + 1, self.height - obj_margin_y)
    
        if obj_type == "circle":
            self.object = {
                "type": "circle",
                "pos": np.array([
                    self.np_random.integers(obj_min_x, obj_max_x),
                    self.np_random.integers(obj_min_y, obj_max_y)
                ], dtype=np.float32),
                "radius": self.width / 25.0,
                "weight": obj_weight,
                "picked": False,
                "broken": False
            }
        else:
            self.object = {
                "type": "square",
                "pos": np.array([
                    self.np_random.integers(obj_min_x, obj_max_x),
                    self.np_random.integers(obj_min_y, obj_max_y)
                ], dtype=np.float32),
                "size": self.width / 14.28,
                "weight": obj_weight,
                "picked": False,
                "broken": False
            }
    
        color_str = "green" if obj_weight == "light" else "red"
        self.mission = f"Pick the {color_str} {self.object['type']} without breaking it. Apply minimum force!"
    
        # Resample object until no initial overlap
        max_resamples = 50  # Prevent infinite loop (rare)
        for _ in range(max_resamples):
            if self.object["type"] == "circle":
                self.object["pos"] = np.array([
                    self.np_random.integers(obj_min_x, obj_max_x),
                    self.np_random.integers(obj_min_y, obj_max_y)
                ], dtype=np.float32)
                dist_to_obj = float(np.linalg.norm(self.agent_pos - self.object["pos"]))
                if dist_to_obj >= (self.agent_radius + self.object["radius"]):
                    break
            else:
                self.object["pos"] = np.array([
                    self.np_random.integers(obj_min_x, obj_max_x),
                    self.np_random.integers(obj_min_y, obj_max_y)
                ], dtype=np.float32)
                dist_to_obj = float(np.linalg.norm(self.agent_pos - self.object["pos"]))
                if dist_to_obj >= (self.object["size"] / 2.0 + self.agent_radius):
                    break
        else:
            # Fallback: move object farther if still overlapping (rare)
            self.object["pos"] = self.agent_pos + np.array([50.0, 50.0], dtype=np.float32)
            self.object["pos"] = np.clip(self.object["pos"], [obj_min_x, obj_min_y], [obj_max_x-1, obj_max_y-1])
    
        self.steps = 0
        self.done = False
        self.last_action_status = None
        
    def _get_description(self):
        weight_str = "light" if self.object["weight"] == "light" else "heavy"
        obj_type_str = self.object["type"]

        if self.object["picked"] or self.object["broken"]:
            return f"Agent sees nothing."

        # Compute relative position
        rel_pos = self.object["pos"] - self.agent_pos
        dist_pixels = float(np.linalg.norm(rel_pos))

        # Normalize distance to action units (using move_scale)
        move_scale = self.width / 30.0
        dist_units = dist_pixels / move_scale

        # Relative bearing (angle from agent's facing direction)
        abs_angle = np.arctan2(rel_pos[1], rel_pos[0])
        rel_angle = abs_angle - self.agent_angle
        # Normalize to [-pi, pi]
        rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))
        bearing_deg = np.degrees(rel_angle)

        loc_desc = f"bearing {bearing_deg:.0f} degrees"
        return f"Agent sees a {weight_str} {obj_type_str} at {loc_desc} which is {dist_units:.1f} units apart."

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_state()
        if self.verbose:
            info = {"mission": self.mission, "description": self._get_description()}
        else:
            info = {"mission": self.mission}
        return self._get_obs(), info

    def step(self, action):
        self.last_action_status = None
        # Map actions from [-1, 1] to original ranges
        mapped_angle_delta_cmd = np.pi * float(action[0])  # [-1, 1] -> [-pi, pi]
        mapped_dist = (float(action[1]) + 1.0) / 2.0       # [-1, 1] -> [0, 1]
        mapped_force = 5.0 * (float(action[2]) + 1.0)      # [-1, 1] -> [0, 10]

        self.steps += 1

        # Movement scale (pixels per unit distance)
        move_scale = self.width / 10
        
        # Compute distance before movement for approach reward
        dist_before = float(np.linalg.norm(self.agent_pos - self.object["pos"]))

        # Rate-limited, incremental heading update (no hard clipping):
        # allowed change grows with forward input, emulating speed-proportional turn rate.
        # allowed = max_turn_rads at dist=1, linearly scaled by current dist in [0,1].
        allowed = float(self.max_turn_rads) * max(0.1, float(mapped_dist))
        cmd = float(mapped_angle_delta_cmd)

        # Soft scale factor: shrink command proportionally if it exceeds allowed magnitude.
        # This yields a smooth, continuous limiter instead of a hard clip.
        scale = min(1.0, allowed / (abs(cmd) + 1e-12))
        applied_delta = cmd * scale

        self.agent_angle += applied_delta

        # Update position using the new heading
        dx = float(mapped_dist) * move_scale * math.cos(self.agent_angle)
        dy = float(mapped_dist) * move_scale * math.sin(self.agent_angle)
        self.agent_pos += np.array([dx, dy], dtype=np.float32)
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.agent_radius, self.width-self.agent_radius)
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.agent_radius, self.height-self.agent_radius)
        
        # Compute distance after movement
        dist_after = float(np.linalg.norm(self.agent_pos - self.object["pos"]))

        reward = -0.05 # Ensure timestep penalty
        terminated = False
        truncated = False
        
        # Add approach reward: positive if closer, negative if farther (shaping)
        dist_change = dist_before - dist_after
        reward += self.approach_reward_coeff * dist_change
        # New: Small per-step penalty for high force (discourages during approach)
        reward -= self.approach_force_penalty * float(mapped_force)

        # Improved interaction
        dist_to_obj = float(np.linalg.norm(self.agent_pos - self.object["pos"]))
        interact = False
        if self.object["type"] == "circle":
            interact = dist_to_obj < (self.agent_radius + self.object["radius"])
        else:
            size = float(self.object["size"])
            half = size / 2.0
            rect_min = self.object["pos"] - half
            rect_max = self.object["pos"] + half
            clamped = np.clip(self.agent_pos, rect_min, rect_max)
            dist_to_rect = float(np.linalg.norm(self.agent_pos - clamped))
            interact = dist_to_rect < self.agent_radius

        if interact and not self.object["picked"] and not self.object["broken"]:
            threshold = self.light_weight_threshold if self.object["weight"] == "light" else self.heavy_weight_threshold
            min_force = threshold * 0.8  # Baseline for successful pick

            if mapped_force >= threshold:
                # Break: High penalty
                self.object["broken"] = True
                reward += -2.0
                terminated = True
                self.last_action_status = "broken"
            elif mapped_force >= min_force:
                # Success: Variable reward based on excess force
                excess = float(mapped_force) - min_force
                variable_reward = 10.0 - (self.force_penalty_coeff * excess)
                # Cap to ensure positive but diminished reward near threshold
                variable_reward = max(variable_reward, 8.0)  # e.g., +8 at max excess
                self.object["picked"] = True
                reward += variable_reward  # Add to any approach penalties
                terminated = True
                self.last_action_status = "picked"
            else:
                # This explicitly tells the agent that this action was a failure.
                reward -= 0.05 # A small but clear penalty
                self.last_action_status = "weak"

        if self.steps >= self.max_steps:
            truncated = True
            # Only apply this penalty if the episode wasn't already terminated by a pick/break
            if not terminated:
                reward -= 2.0 # A penalty for running out of time
            
        if self.verbose:
            info = {"mission": self.mission, "description": self._get_description()}
        else:
            info = {"mission": self.mission}

        return self._get_obs(), reward, terminated, truncated, info
    
    def getframe(self):
        """Always returns the current rendered image, regardless of observation mode."""
        return self._get_image_obs()
    
    def _get_features(self):
        """Compiles and returns the environment state as a feature vector."""
        # Agent features
        agent_angle_sin = np.sin(self.agent_angle)
        agent_angle_cos = np.cos(self.agent_angle)

        # Object features (relative to agent)
        rel_pos = (self.object["pos"] - self.agent_pos)
        rel_pos_x_norm = rel_pos[0] / self.width
        rel_pos_y_norm = rel_pos[1] / self.height

        # Relative bearing
        abs_angle_to_obj = np.arctan2(rel_pos[1], rel_pos[0])
        rel_angle = abs_angle_to_obj - self.agent_angle
        rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle)) # Normalize to [-pi, pi]
        rel_angle_sin = np.sin(rel_angle)
        rel_angle_cos = np.cos(rel_angle)

        # One-hot encodings for object type and weight
        is_circle = 1.0 if self.object["type"] == "circle" else 0.0
        is_square = 1.0 - is_circle
        is_light = 1.0 if self.object["weight"] == "light" else 0.0
        is_heavy = 1.0 - is_light
        
        # Status flags
        is_picked = 1.0 if self.object["picked"] else 0.0
        is_broken = 1.0 if self.object["broken"] else 0.0
        
        # If object is gone, zero out its relative info
        if is_picked or is_broken:
            rel_pos_x_norm = 0.0
            rel_pos_y_norm = 0.0
            rel_angle_sin = 0.0
            rel_angle_cos = 0.0

        features = np.array([
            agent_angle_sin,
            agent_angle_cos,
            rel_pos_x_norm,
            rel_pos_y_norm,
            rel_angle_sin,
            rel_angle_cos,
            is_circle,
            is_square,
            is_light,
            is_heavy,
            is_picked,
            is_broken
        ], dtype=np.float32)

        return features
    
    def _get_image_obs(self):
        """Renders and returns the environment state as an RGB image array."""
        W, H = self.width, self.height
        if self.ssaa_scale > 1:
            big_w, big_h = W * self.ssaa_scale, H * self.ssaa_scale
            big = pygame.Surface((big_w, big_h), flags=pygame.SRCALPHA)
            self._draw_scene(big, scale=self.ssaa_scale)
            small = pygame.transform.smoothscale(big, (W, H))
            arr = np.transpose(np.array(pygame.surfarray.pixels3d(small), copy=True), (1, 0, 2))
        else:
            surface = pygame.Surface((W, H), flags=pygame.SRCALPHA)
            self._draw_scene(surface, scale=1)
            arr = np.transpose(np.array(pygame.surfarray.pixels3d(surface), copy=True), (1, 0, 2))
        return arr.astype(np.uint8)
    
    def render_observation(self, obs):
        """Renders the given observation as an RGB image."""
        if self.observation_mode == "images":
            # If observation is already an image, return it
            return obs
        else:
            features = obs
            agent_angle = np.arctan2(features[0], features[1])
            rel_pos = np.array([features[2] * self.width, features[3] * self.height])
            obj_type = "circle" if features[6] > 0.5 else "square"
            obj_weight = "light" if features[8] > 0.5 else "heavy"
            picked = features[10] > 0.5
            broken = features[11] > 0.5
            agent_pos = np.array([self.width / 2.0, self.height / 2.0])
            obj_pos = agent_pos + rel_pos
            object_dict = {
                "type": obj_type,
                "pos": obj_pos,
                "weight": obj_weight,
                "picked": picked,
                "broken": broken
            }
            if obj_type == "circle":
                object_dict["radius"] = self.width / 25.0
            else:
                object_dict["size"] = self.width / 14.28
            last_action_status = None
            W, H = self.width, self.height
            if self.ssaa_scale > 1:
                big_w, big_h = W * self.ssaa_scale, H * self.ssaa_scale
                big = pygame.Surface((big_w, big_h), flags=pygame.SRCALPHA)
                big.fill((100, 100, 100))
                PickEnv.draw_scene(big, self.ssaa_scale, agent_pos, agent_angle, self.agent_radius, self.agent_shape, object_dict, last_action_status)
                small = pygame.transform.smoothscale(big, (W, H))
                arr = np.transpose(np.array(pygame.surfarray.pixels3d(small), copy=True), (1, 0, 2))
            else:
                surface = pygame.Surface((W, H), flags=pygame.SRCALPHA)
                surface.fill((100, 100, 100))
                PickEnv.draw_scene(surface, 1, agent_pos, agent_angle, self.agent_radius, self.agent_shape, object_dict, last_action_status)
                arr = np.transpose(np.array(pygame.surfarray.pixels3d(surface), copy=True), (1, 0, 2))
            return arr.astype(np.uint8)

    # ---------------------------
    # Rendering helpers
    # ---------------------------
    def _draw_scene(self, surface, scale=1):
        """Draw environment onto given surface at coordinate scale factor."""
        # Background
        surface.fill((100, 100, 100))

        # Draw object - green for light, red for heavy
        color = (0, 255, 0) if self.object["weight"] == "light" else (255, 0, 0)
        if not self.object["broken"]:
            if self.object["type"] == "circle":
                cx = int(self.object["pos"][0] * scale)
                cy = int(self.object["pos"][1] * scale)
                r = int(self.object["radius"] * scale)
                gfxdraw.filled_circle(surface, cx, cy, r, color)
                gfxdraw.aacircle(surface, cx, cy, r, color)
            else:
                size = int(self.object["size"] * scale)
                x = int(self.object["pos"][0] * scale)
                y = int(self.object["pos"][1] * scale)
                # Draw square as filled AA polygon for crisp edges
                left = x - size // 2
                top = y - size // 2
                pts = [
                    (left, top),
                    (left + size, top),
                    (left + size, top + size),
                    (left, top + size),
                ]
                gfxdraw.filled_polygon(surface, pts, color)
                gfxdraw.aapolygon(surface, pts, color)
                
        # ------------------------
        # Draw full-frame bounding box (status-based)
        # ------------------------
        box_color = None
        if self.last_action_status == "picked":
            box_color = (0, 255, 0)   # green
        elif self.last_action_status == "broken":
            box_color = (255, 0, 0)   # red
        elif self.last_action_status == "weak":
            box_color = (0, 0, 255)   # blue

        if box_color is not None:
            surf_w, surf_h = surface.get_width(), surface.get_height()
            thickness = max(1, int(round(3 * scale)))
            pygame.draw.rect(surface, box_color, pygame.Rect(0, 0, surf_w, surf_h), thickness)

        # Draw agent
        if self.agent_shape == "disk":
            self._draw_agent_disk(surface, scale)
        else:
            self._draw_agent_triangle(surface, scale)

    def _draw_agent_triangle(self, surface, scale=1):
        ax = float(self.agent_pos[0] * scale)
        ay = float(self.agent_pos[1] * scale)
        r = float(self.agent_radius * scale)
        ang = float(self.agent_angle)

        # Use a clear, standard angle for the base (e.g., 135 degrees)
        base_angle = 3 * math.pi / 4

        # --- 1. Calculate and draw the main red triangle ---
        tip = (int(ax + r * math.cos(ang)), int(ay + r * math.sin(ang)))
        left = (int(ax + r * math.cos(ang + base_angle)), int(ay + r * math.sin(ang + base_angle)))
        right = (int(ax + r * math.cos(ang - base_angle)), int(ay + r * math.sin(ang - base_angle)))

        main_pts = [tip, left, right]
        # Draw the main red body of the agent
        gfxdraw.filled_polygon(surface, main_pts, (255, 0, 0))
        gfxdraw.aapolygon(surface, main_pts, (255, 0, 0))

        # --- 2. Calculate and draw the smaller white tip triangle ---
        # This factor determines the size of the tip (e.g., 25% of the main triangle's edge length)
        tip_scale_factor = 0.50

        # The new base points for the tip triangle are found by moving from the
        # main 'tip' towards the main 'left' and 'right' corners.
        tip_left_x = int(tip[0] * (1 - tip_scale_factor) + left[0] * tip_scale_factor)
        tip_left_y = int(tip[1] * (1 - tip_scale_factor) + left[1] * tip_scale_factor)

        tip_right_x = int(tip[0] * (1 - tip_scale_factor) + right[0] * tip_scale_factor)
        tip_right_y = int(tip[1] * (1 - tip_scale_factor) + right[1] * tip_scale_factor)

        # The tip triangle consists of the main tip and the two new points
        tip_pts = [
            tip, 
            (tip_left_x, tip_left_y),
            (tip_right_x, tip_right_y)
        ]

        # Draw the white tip on top of the main triangle
        gfxdraw.filled_polygon(surface, tip_pts, (255, 255, 255))
        gfxdraw.aapolygon(surface, tip_pts, (255, 255, 255))
        
    def _draw_agent_disk(self, surface, scale=1):
        # Body
        ax = int(self.agent_pos[0] * scale)
        ay = int(self.agent_pos[1] * scale)
        r = int(self.agent_radius * scale)
        gfxdraw.filled_circle(surface, ax, ay, r, (255, 0, 0))
        gfxdraw.aacircle(surface, ax, ay, r, (255, 0, 0))

        # Heading tick (AA line)
        ang = float(self.agent_angle)
        hx = int(ax + r * math.cos(ang))
        hy = int(ay + r * math.sin(ang))
        pygame.draw.aaline(surface, (255, 255, 255), (ax, ay), (hx, hy))

    # ---------------------------
    # Observation and render
    # ---------------------------
    def _get_obs(self):
        """Dispatcher to return observation based on the configured mode."""
        if self.observation_mode == "features":
            return self._get_features()
        else: # "image"
            return self._get_image_obs()

    def render(self):
        if self.render_mode != "human":
            return self._get_image_obs()

        # Lazy-init display
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.render_scale, self.height * self.render_scale))
            self.clock = pygame.time.Clock()

        # Draw once at max(scale for quality)
        draw_scale = max(self.ssaa_scale, self.render_scale)
        big_w, big_h = self.width * draw_scale, self.height * draw_scale
        big = pygame.Surface((big_w, big_h), flags=pygame.SRCALPHA)
        # Use a slightly lighter bg for window
        big.fill((200, 200, 200))
        self._draw_scene(big, scale=draw_scale)

        # Scale to window
        target_size = (self.width * self.render_scale, self.height * self.render_scale)
        display_img = pygame.transform.smoothscale(big, target_size)
        self.screen.blit(display_img, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            
def save_dataset_for_diffusers(dataset, save_dir):
    """
    Saves a dataset of images and captions in the format expected by diffusers.

    This creates a directory with an 'images' subfolder and a 'metadata.jsonl' file.

    Args:
        dataset (list): A list of dictionaries, where each dict has "frame" and "description".
        save_dir (str): The path to the root directory where the dataset will be saved.
    """
    if not dataset:
        print("Warning: No data to save.")
        return
        
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    metadata_entries = []
    
    print(f"Saving {len(dataset)} items to {save_dir}...")
    for i, item in enumerate(dataset):
        try:
            # Create a Pillow Image from the numpy array
            img = Image.fromarray(item["frame"])
            
            # Define image filename and save it
            base_filename = f"{i:06d}.png"
            img_path = os.path.join(images_dir, base_filename)
            img.save(img_path)
            
            # Create metadata entry
            # The file_name must be relative to the root of the dataset directory
            metadata_entry = {
                "file_name": os.path.join("images", base_filename),
                "text": item["description"]
            }
            metadata_entries.append(metadata_entry)

        except Exception as e:
            print(f"Error processing item {i}: {e}")

    # Write the metadata.jsonl file
    metadata_path = os.path.join(save_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Successfully saved dataset with {len(metadata_entries)} entries.")
    print(f"Images saved in: {images_dir}")
    print(f"Metadata saved in: {metadata_path}")
    
def save_feature_dataset(dataset, save_dir):
    """
    Saves a dataset of feature vectors and captions in a structure
    compatible with dataset loaders.

    This creates a directory with a 'features' subfolder containing .npy files
    and a 'metadata.jsonl' file at the root.

    Args:
        dataset (list): A list of dictionaries, where each dict has "frame" (a numpy array)
                        and "description".
        save_dir (str): The path to the root directory where the dataset will be saved.
    """
    if not dataset:
        print("Warning: No data to save.")
        return
        
    features_dir = os.path.join(save_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    metadata_entries = []
    
    print(f"Saving {len(dataset)} feature vectors to {save_dir}...")
    for i, item in enumerate(dataset):
        try:
            feature_vector = item["frame"]
            
            # Define numpy filename and save it
            base_filename = f"{i:06d}.npy"
            feature_path = os.path.join(features_dir, base_filename)
            np.save(feature_path, feature_vector)
            
            # Create metadata entry. The file_name must be relative to the root of the dataset directory.
            metadata_entry = {
                "file_name": os.path.join("features", base_filename),
                "text": item["description"]
            }
            metadata_entries.append(metadata_entry)

        except Exception as e:
            print(f"Error processing item {i}: {e}")

    # Write the metadata.jsonl file
    metadata_path = os.path.join(save_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Successfully saved dataset with {len(metadata_entries)} entries.")
    print(f"Feature .npy files saved in: {features_dir}")
    print(f"Metadata saved in: {metadata_path}")
            
@hydra.main(version_base=None, config_path="../config/env", config_name="PickEnv")
def main(cfg: DictConfig) -> None:
    is_collect_data = True
    cfg.observation_mode = "features"
    if is_collect_data:
        mode="collect_data"
    else:
        mode="train"
    cfg.verbose = True
    cfg.max_steps = 50
    env = PickEnv(cfg, mode)
    paired_data = []
    # Total number of timesteps to collect
    total_training_data = 150000
    validation_data = 100
    paired_data, episode = collect_data(env, total_training_data + validation_data)
    env.close()
    
    # --- Dataset Saving Logic ---
    if is_collect_data:
        training_data = paired_data[:total_training_data]
        val_data = paired_data[total_training_data:]

        # --- DYNAMIC SAVING BASED ON OBSERVATION MODE ---
        if cfg.observation_mode == "images":
            save_dir_train = f"data/{env.name}/training_images"
            print("\n--- Saving Image Training Dataset ---")
            save_dataset_for_diffusers(training_data, save_dir_train)

            save_dir_val = f"data/{env.name}/validation_images"
            print("\n--- Saving Image Validation Dataset ---")
            save_dataset_for_diffusers(val_data, save_dir_val)

        elif cfg.observation_mode == "features":
            # Call the new numpy saving function
            save_dir_train = f"data/{env.name}/training_features"
            print("\n--- Saving Feature Training Dataset (NumPy format) ---")
            save_feature_dataset(training_data, save_dir_train)

            save_dir_val = f"data/{env.name}/validation_features"
            print("\n--- Saving Feature Validation Dataset (NumPy format) ---")
            save_feature_dataset(val_data, save_dir_val)


        
if __name__ == "__main__":
    main()