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
from sac_agent.buffer import ReplayBuffer
from helper_functions.utils import wrap_env
from omegaconf import DictConfig, OmegaConf
from successor_feature.get_target_weight import TransferEvaluation


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
        env_name = cfg.env
        #Load environment
        if env_name == "SimplePickup":
            #Source Env
            from env.env import MultiObjectMiniGridEnv, FlattenedFrameStack
            from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
            from gymnasium.wrappers import FrameStack
            source_env = MultiObjectMiniGridEnv(max_steps=20, agent_view_size=5, size=7, to_be_terminated=True)
            source_env = wrap_env(source_env, [(RGBImgPartialObsWrapper, None), (ImgObsWrapper, None), (FrameStack, {"num_stack":cfg.frame_stack}), (FlattenedFrameStack, None)])
            target_env1 = MultiObjectMiniGridEnv(max_steps=20, agent_view_size=5, size=7, to_be_terminated=True, is_single_object=True, reward_object="Red_ball")
            target_env1 = wrap_env(target_env1, [(RGBImgPartialObsWrapper, None), (ImgObsWrapper, None), (FrameStack, {"num_stack":cfg.frame_stack}), (FlattenedFrameStack, None)])
            target_env2 = MultiObjectMiniGridEnv(max_steps=20, agent_view_size=5, size=7, to_be_terminated=True, is_single_object=False, reward_object="Red_ball")
            target_env2 = wrap_env(target_env2, [(RGBImgPartialObsWrapper, None), (ImgObsWrapper, None), (FrameStack, {"num_stack":cfg.frame_stack}), (FlattenedFrameStack, None)])
            target_env3 = MultiObjectMiniGridEnv(max_steps=20, agent_view_size=5, size=7, to_be_terminated=True, is_single_object=False, reward_object="Both")
            target_env3 = wrap_env(target_env3, [(RGBImgPartialObsWrapper, None), (ImgObsWrapper, None), (FrameStack, {"num_stack":cfg.frame_stack}), (FlattenedFrameStack, None)])
            self.envs = [source_env, target_env1, target_env2, target_env3]
            self.interaction_step = [cfg.num_train_frames, cfg.transfer_interaction_step, cfg.transfer_interaction_step, cfg.transfer_interaction_step]
            self.num_seed_frames = [cfg.num_seed_frames, cfg.batch_size, cfg.batch_size, cfg.batch_size]

        elif env_name == "3DRoom":
            from env.miniworld import ColorBoxesEnv
            self.env = ColorBoxesEnv(max_episode_steps=math.inf, floor_type='asphalt')

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

        # create logger
        if cfg.use_wandb:
            wandb.init(project="SuccessorFeature", name=self.agent.__class__.__name__, config=OmegaConf.to_container(cfg, resolve=True))
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
                                            meta["task"],
                                            self.global_step,
                                            eval_mode=False,
                                            epsilon = self.epsilon
                                            )
    
                # try to update the agent
                if not seed_until_step(task_step):
                    metrics = self.agent.update(reply_buffer, self.global_step)
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
                        wandb.log({f"train/episode_reward": episode_reward,
                                   f"train/success" : success,
                                   f"train/episode" : self._global_episode}, step = self.global_step)
                    observation, _ = env.reset()
                    episode_step = 0
                    episode_reward = 0
                    success = 0
                    if self._global_step > self.cfg.epsilon_decay_after:
                        self.update_epsilon()
    
                if eval_every_step(self._global_step):
                    avg_reward = self.evaluate_agent(env, task_step)
    
                if save_every_step(self._global_step):
                    self.save_snapshot()
    
                if self.cfg.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step = self.global_step)
            avg_reward, green_ball_picked, red_ball_picked = self.evaluate_agent(env, task_step)
            if i == 0 :
                self.source_agent = self.agent
            self.save_snapshot()
            print("[INFO] Average reward: ", avg_reward)
            print("[INFO] Green ball picked: ", green_ball_picked)
            print("[INFO] Red ball picked: ", red_ball_picked)
            print("-----------[INFO] Training on env: ", env.env_name, " is done-------------")
        return self.source_agent

    def evaluate_agent(self, env, task_step):
        """Runs evaluation episodes and logs average reward to wandb."""
        green_ball_picked = 0
        red_ball_picked = 0
        total_reward = 0
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
                if env.picked == "Green_ball":
                    green_ball_picked += 1
                if env.picked == "Red_ball":
                    red_ball_picked += 1
            total_reward += episode_reward
        avg_reward = total_reward / self.cfg.num_eval_episodes
        # print(f"[INFO] Evaluation reward of {env.env_name} at step {task_step}: ", avg_reward)
        if self.cfg.use_wandb:
            wandb.log({f"eval/avg_reward": avg_reward}, step = self.global_step)
        return avg_reward, green_ball_picked, red_ball_picked

    def update_epsilon(self):
        """
        Exponentially decays epsilon.
        """
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

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
    target = [("Red_ball", True), ("Red_ball", False), ("Both", False)]
    for goal, is_single_object in target:
        if cfg.env == "SimplePickup":
            from env.env import MultiObjectMiniGridEnv, FlattenedFrameStack
            target_env = MultiObjectMiniGridEnv(max_steps=20, agent_view_size=5, size=7, to_be_terminated=True, reward_object=goal, is_single_object=is_single_object)
            from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
            from gymnasium.wrappers import FrameStack
            target_env = RGBImgPartialObsWrapper(target_env)
            target_env = ImgObsWrapper(target_env)
            target_env = FrameStack(target_env, cfg.frame_stack)
            target_env = FlattenedFrameStack(target_env)

        elif cfg.env == "3DRoom":
            from env.miniworld import ColorBoxesEnv
            target_env = ColorBoxesEnv(max_episode_steps=math.inf, floor_type='asphalt')

        transfer.get_transfer_result(target_env, goal, is_single_object, cfg.transfer_interaction_step, cfg.num_eval_episodes)

    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()