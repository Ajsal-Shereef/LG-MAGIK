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
from buffer import ReplayBuffer
from helper_functions.utils import load_config
from omegaconf import DictConfig, OmegaConf
from successor_feature.get_target_weight import TransferEvaluation


def make_agent(obs_type, obs_spec_shape, action_dim, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec_shape
    cfg.action_shape = action_dim
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

class Trainer():
    def __init__(self, cfg):

        self._global_step = 0
        self._global_episode = 0
        self.cfg = cfg
        env_name = cfg.env

        #Load environment
        if env_name == "reacher":
            from env.reacher import Reacher4TargetsEnv
            env_config = load_config("config/reacher.yml")
            source_env = Reacher4TargetsEnv(goal_target_idx=env_config["goal_target_idx"], max_steps = env_config["max_steps"], reaching_threshold=env_config["reaching_threshold"])
            target_env1 = Reacher4TargetsEnv(goal_target_idx=0, max_steps = env_config["max_steps"], reaching_threshold=env_config["reaching_threshold"])
            target_env2 = Reacher4TargetsEnv(goal_target_idx=1, max_steps = env_config["max_steps"], reaching_threshold=env_config["reaching_threshold"])
            target_env3 = Reacher4TargetsEnv(goal_target_idx=3, max_steps = env_config["max_steps"], reaching_threshold=env_config["reaching_threshold"])
            self.envs = [source_env, target_env1, target_env2, target_env3]
            self.interaction_step = [cfg.num_train_frames, cfg.transfer_interaction_step, cfg.transfer_interaction_step, cfg.transfer_interaction_step]
            self.num_seed_frames = [cfg.num_seed_frames, cfg.batch_size, cfg.batch_size, cfg.batch_size]
        else:
            raise NotImplementedError("Target environment not implemented")

        action_dim = self.envs[0].action_space.shape

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            self.envs[0].observation_space.shape,
            action_dim,
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )

        print("[INFO] Agent created: ", self.agent.__class__.__name__)


        # create logger
        if cfg.use_wandb:
            wandb.init(project="SuccessorFeatureReacher", name=self.agent.__class__.__name__, config=OmegaConf.to_container(cfg, resolve=True))
            # wandb.watch(self.agent.encoder, log="all")
            # wandb.watch(self.agent.featureNet, log="all")
            # wandb.watch(self.agent.critic, log="all")
    
    def train(self):
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
                                            meta,
                                            self.global_step,
                                            eval_mode=False,
                                            )

                # try to update the agent
                if not seed_until_step(task_step):
                    metrics = self.agent.update(reply_buffer, self.global_step)
                    metrics["episode"] = self.global_episode
                    metrics["task_step"] = task_step
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
                        wandb.log({"train/episode_reward": episode_reward,
                                   f"train/success" : success,
                                   f"train/episode" : self._global_episode}, step = self.global_step)
                    observation, _ = env.reset()
                    episode_step = 0
                    episode_reward = 0
                    success = 0

                if eval_every_step(self._global_step):
                    avg_reward = self.evaluate_agent(env)

                if save_every_step(self._global_step):
                    self.save_snapshot()

                if self.cfg.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=self.global_step)
            avg_reward, success = self.evaluate_agent(env)
            if i == 0 :
                self.source_agent = self.agent
            self.save_snapshot()
            print("[INFO] Average reward: ", avg_reward)
            print("[INFO] Number of times arms reached the target: ", success)
            print("-----------[INFO] Training on env: ", env.env_name, " is done-------------")
        return self.agent

    def evaluate_agent(self, env):
        """Runs evaluation episodes and logs average reward to wandb."""
        total_reward = 0
        env.is_test = True
        success = 0
        for _ in range(self.cfg.num_eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, meta=self.agent.solved_meta, step=self.global_step, eval_mode=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                if terminated:
                    success += 1
            total_reward += episode_reward
        avg_reward = total_reward / self.cfg.num_eval_episodes
        if self.cfg.use_wandb:
            wandb.log({"eval/avg_reward": avg_reward}, step=self.global_step)
            wandb.log({"eval/success": success}, step=self.global_step)
        env.is_test = False
        return avg_reward, success

    def save_snapshot(self):
        snapshot_dir = Path(self.cfg.snapshot_dir)

        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
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

@hydra.main(config_path=".", config_name="train", version_base=None)
def main(cfg):
    trainer = Trainer(cfg)
    agent = trainer.train()
    transfer = TransferEvaluation(agent, agent.device)
    #initiliase target_env
    #Load environment
    target_class = [0,1,3]
    for tar_class in target_class:
        if cfg.env == "reacher":
            from env.reacher import Reacher4TargetsEnv
            env_config = load_config("config/reacher.yml")
            target_env = Reacher4TargetsEnv(goal_target_idx=tar_class, max_steps = env_config["max_steps"], reaching_threshold=env_config["reaching_threshold"], render_mode = 'rgb_array')

        else:
            raise NotImplementedError("Target environment not implemented")

        transfer.get_transfer_result(target_env, tar_class, cfg.transfer_interaction_step, 20)

    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()