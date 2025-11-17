import utils
import torch
import numpy as np

import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo

class TransferEvaluation():
    def __init__(self, agent, device, transform):
        self.agent = agent
        self.device = device
        self.transform = transform
        
    def collect_target_data(self, env, target_interaction_steps):
        step = 0
        obs, _ = env.reset()
        phi_values_target = []
        rewards_target = []
        while step < target_interaction_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            phi = self.agent.get_phi(obs.to(self.device), torch.tensor(action).unsqueeze(0).to(self.device))
            phi_values_target.append(phi.squeeze().detach().cpu().numpy())
            rewards_target.append(reward)
            obs = next_obs
            step += 1
            if done:
                obs, _ = env.reset()
        return phi_values_target, rewards_target

    def test_transfer(self, env, w, target_episodes):
        episode = 0
        obs, _ = env.reset()
        while episode < target_episodes:
            done = False
            success = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, w.astype(np.float32), 0, True, 0)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obs = next_obs
                if done:
                    if terminated:
                        success += 1
                    obs, _ = env.reset()
                    episode += 1

        env.close()
        if env.name == "SimplePickup":
            agent_performance = env.unwrapped.get_performance_metric()
        else:
            agent_performance = env.get_performance_metric()
        print(f"[INFO] Agent performance: ", agent_performance)
    
    def get_transfer_result(self, env, target_interaction_steps, target_episodes):
        phi_values_target, reward_target = self.collect_target_data(env, target_interaction_steps)
        w = np.linalg.lstsq(phi_values_target, reward_target, rcond=None)[0]
        print("[INFO] Target weight max value: ", np.max(w))
        print("[INFO] Target weight min value: ", np.min(w))
        # target_env = RecordVideo(env, f"videos/{self.agent.__class__.__name__}/{goal}_{is_single_object}", episode_trigger=lambda x: True, disable_logger=True)
        self.test_transfer(env, w, target_episodes)
        print(f"[INFO] Target experimnet {env.env_name} is done")
        print("--------------------------------------------------------------------------")