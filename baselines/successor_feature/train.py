import math
import torch
import hydra
import wandb
import random
import numpy as np
import torch.nn as nn

import utils

import sys
sys.path.append('.')

from pathlib import Path
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from agent.agent_utils.buffer import ReplayBuffer
from baselines.successor_feature.get_target_weight import TransferEvaluation
from architectures.common_utils import get_train_transform_cnn, get_train_transform_mlp, create_dump_directory


def make_agent(obs_type, obs_spec_shape, action_dim, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec_shape
    cfg.action_shape = action_dim
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

def update_epsilon(agent):
    """
    Decays epsilon linearly over time.
    """
    agent.epsilon_step += 1
    decay_ratio = min(agent.epsilon_step / agent.epsilon_decay, 1.0)
    agent.epsilon = agent.epsilon_end + (1.0 - agent.epsilon_end) * (1.0 - decay_ratio)

class Trainer():
    def __init__(self, cfg):

        self._global_step = 0
        self._global_episode = 0
        self.cfg = cfg
        self.env_name = cfg.env.name
        self.envs = []
        #Load environment
        if self.env_name == "PickEnv":
            from env.PickEnv import PickEnv
            env = PickEnv(cfg.env, mode=cfg.env.mode, render_mode="rgb_array")
        elif self.env_name == "MiniWorld":
            from env.MiniWorld import PickObjectEnv
            from architectures.common_utils import SwitchChannel
            env_0 = SwitchChannel(PickObjectEnv(cfg.env))
            self.envs.append(env_0)
            cfg.env.objects = [["duckie", "ball"]]
            cfg.env.reward_objects = [["duckie"]]
            env_1 = SwitchChannel(PickObjectEnv(cfg.env))
            self.envs.append(env_1)
            cfg.env.layout = ["wood/brick_wall"]
            env_2 = SwitchChannel(PickObjectEnv(cfg.env))
            self.envs.append(env_2)
            cfg.env.objects = [["duckie", "box"]]
            cfg.env.reward_objects = [["duckie"]]
            cfg.env.layout = ["grass/concrete"]
            env_3 = SwitchChannel(PickObjectEnv(cfg.env))
            self.envs.append(env_3)
        elif self.env_name ==  "SimplePickup":
            from env.SimplePickup import SimplePickup
            from minigrid.wrappers import RGBImgPartialObsWrapper
            from minigrid.wrappers import ImgObsWrapper
            from architectures.common_utils import SwitchChannel
            env1 = SimplePickup(cfg.env)
            env1 = RGBImgPartialObsWrapper(env1, tile_size=cfg.env.tile_size)
            env1 = SwitchChannel(ImgObsWrapper(env1))
            self.envs.append(env1)
            cfg.env.objects = [["purple box", "red ball"]]
            cfg.env.reward_objects = [["purple box"]]
            env2 = SimplePickup(cfg.env)
            env2 = RGBImgPartialObsWrapper(env2, tile_size=cfg.env.tile_size)
            env2 = SwitchChannel(ImgObsWrapper(env2))
            self.envs.append(env2)
            cfg.env.objects = [["purple box", "green ball"]]
            cfg.env.reward_objects = [["purple box"]]
            env3 = SimplePickup(cfg.env)
            env3 = RGBImgPartialObsWrapper(env3, tile_size=cfg.env.tile_size)
            env3 = SwitchChannel(ImgObsWrapper(env3))
            self.envs.append(env3)
            
        self.interaction_step = [cfg.num_train_frames]*3
        # self.interaction_step = [cfg.num_train_frames, int(cfg.num_train_frames*0.10), int(cfg.num_train_frames*0.10)]
        self.num_seed_frames = [cfg.num_seed_frames]*3

        action_dim = (int(self.envs[0].action_space.n),)

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            self.envs[0].observation_space.shape,
            action_dim,
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )

        print("[INFO] Agent created: ", self.agent.__class__.__name__)

        self.epsilon = self.cfg.epsilon_start
        
        # Save model directory
        self.snapshot_dir = create_dump_directory(self.cfg.snapshot_dir)

        # create logger
        if cfg.use_wandb:
            wandb.init(project="SuccessorFeature_p_3", name=f"{self.agent.__class__.__name__}_{self.env_name}", config=OmegaConf.to_container(cfg, resolve=True))
    
    def train(self, env_name):
        reply_buffer = ReplayBuffer(buffer_size=self.cfg.replay_buffer_size, batch_size=self.cfg.batch_size, device=self.cfg.device)
        for i, (env, interaction_step, num_seed_frames) in enumerate(zip(self.envs, self.interaction_step, self.num_seed_frames)):
            train_until_step = utils.Until(interaction_step, 1)
            seed_until_step = utils.Until(num_seed_frames, 1)
            eval_every_step = utils.Every(self.cfg.eval_every_frames, 1)
            save_every_step = utils.Every(self.cfg.save_checkpoint_every, 1)
            reply_buffer.clear()
            task_step = 0
            episode_step, episode_reward, success = 0, 0, 0
            observation, info = env.reset()
            while train_until_step(task_step + 1):
            
                if seed_until_step(self.global_step):
                    meta = self.agent.init_meta()
                else:
                    meta = self.agent.solved_meta
    
                # sample action
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(observation,
                                            meta["task"],
                                            self.global_step,
                                            eval_mode=False,
                                            epsilon = self.epsilon
                                            )
    
                # try to update the agent
                if not seed_until_step(task_step):
                    metrics = self.agent.update(reply_buffer, self.global_step)
                    metrics["task_step"] = task_step
                    metrics["Steps"] = self._global_step
                else:
                    metrics = dict()
                    
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                reply_buffer.add(observation, action, reward, next_observation, done, meta["task"])
                observation = next_observation
                episode_reward += reward
                episode_step += 1
                task_step += 1
                self._global_step += 1
    
                if terminated:
                    success += 1
    
                if done:
                    self._global_episode += 1
                    if self.cfg.use_wandb:
                        wandb.log({f"train/episode_reward": episode_reward,
                                   f"train/success" : success,
                                   f"train/episode" : self._global_episode,
                                   f"train/Steps" : self._global_episode})
                    observation, _ = env.reset()
                    episode_step = 0
                    episode_reward = 0
                    success = 0
                    if self._global_step > self.cfg.epsilon_decay_after:
                        self.update_epsilon()
    
                # if eval_every_step(self._global_step):
                #     avg_reward = self.evaluate_agent(env, task_step)
    
                # if save_every_step(self._global_step):
                #     self.save_snapshot()
    
                if self.cfg.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()})
            print("-----------[INFO] Final evaluation after training---------------")
            self.evaluate_agent(env, task_step)
            if i == 0 :
                self.source_agent = self.agent
            # self.save_snapshot()
            print("-----------[INFO] Training on env: ", env.unwrapped.env_name, " is done-------------")
        return self.source_agent

    def evaluate_agent(self, env, task_step):
        """Runs evaluation episodes and logs average reward to wandb."""
        if self.env_name == "SimplePickup" or self.env_name == "MiniWorld":
            env.unwrapped.reset_metrices()
        else:
            env.reset_metrices()
        for _ in range(self.cfg.num_eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, meta=self.agent.solved_meta["task"], step=self.global_step, eval_mode=True, epsilon=0.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
        if self.env_name ==  "SimplePickup" or self.env_name ==  "MiniWorld":
            agent_performance = env.unwrapped.get_performance_metric()
        else:
            agent_performance = env.get_performance_metric()
        # if self.cfg.use_wandb:
        #     wandb.log(agent_performance)
        print(f"Agent performance after {task_step}: ", agent_performance)

    def update_epsilon(self):
        """
        Exponentially decays epsilon.
        """
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def save_snapshot(self):
        snapshot = Path(f"{self.snapshot_dir}/snapshot_{self.global_frame}.pt")
        keys_to_save = [
            "agent",
            "_global_step",
            "_global_episode"
        ]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)
            print(f"[INFO] snapshot saved to {snapshot}")
            
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

@hydra.main(config_path="../../config", config_name="successor_feature_config", version_base=None)
def main(cfg):
    trainer = Trainer(cfg)
    agent = trainer.train(cfg.env.name)
    # Create the transfer environment
    transfer_env = []
    env_name = cfg.env.name
    if env_name == "PickEnv":
        from env.PickEnv import PickEnv
        env = PickEnv(cfg.env, mode=cfg.env.mode, render_mode="rgb_array")
        train_transform = get_train_transform_mlp
    elif env_name == "MiniWorld":
        from env.MiniWorld import PickObjectEnv
        from architectures.common_utils import SwitchChannel
        cfg.env.objects = [["duckie", "ball"]]
        cfg.env.reward_objects = [["duckie"]]
        env_1 = SwitchChannel(PickObjectEnv(cfg.env))
        transfer_env.append(env_1)
        cfg.env.layout = ["wood/brick_wall"]
        env_2 = SwitchChannel(PickObjectEnv(cfg.env))
        transfer_env.append(env_2)
        cfg.env.objects = [["duckie", "box"]]
        cfg.env.reward_objects = [["duckie"]]
        cfg.env.layout = ["grass/concrete"]
        env_3 = SwitchChannel(PickObjectEnv(cfg.env))
        transfer_env.append(env_3)
    elif env_name ==  "SimplePickup":
        from env.SimplePickup import SimplePickup
        from minigrid.wrappers import RGBImgPartialObsWrapper
        from minigrid.wrappers import ImgObsWrapper
        from architectures.common_utils import SwitchChannel
        cfg.env.objects = [["purple box", "red ball"]]
        cfg.env.reward_objects = [["purple box"]]
        env1 = SimplePickup(cfg.env)
        env1 = RGBImgPartialObsWrapper(env1, tile_size=cfg.env.tile_size)
        env1 = SwitchChannel(ImgObsWrapper(env1))
        transfer_env.append(env1)
        cfg.env.objects = [["purple box", "green ball"]]
        cfg.env.reward_objects = [["purple box"]]
        env2 = SimplePickup(cfg.env)
        env2 = RGBImgPartialObsWrapper(env2, tile_size=cfg.env.tile_size)
        env2 = SwitchChannel(ImgObsWrapper(env2))
        transfer_env.append(env2)
        # cfg.env.wall_color = ["blue"]
        # env3 = SimplePickup(cfg.env)
        # env3 = RGBImgPartialObsWrapper(env3, tile_size=cfg.env.tile_size)
        # env3 = SwitchChannel(ImgObsWrapper(env3))
        # transfer_env.append(env3)
    
    transfer = TransferEvaluation(agent, agent.device, env_name)
    for env in transfer_env:
        transfer.get_transfer_result(env, cfg.transfer_interaction_step, cfg.num_eval_episodes)

    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()