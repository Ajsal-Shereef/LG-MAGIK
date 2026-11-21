import torch
import wandb
import warnings
import argparse
import gymnasium
import numpy as np
from collections import deque
from helper_functions.utils import *
import random
from sac_agent.agent import SAC

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--env", type=str, default="3DRoom", help="Gym environment name, default: SimplePickup")
    parser.add_argument("--env_reward_object", type=str, default="Green_ball", help="Which object reward in the env Green_ball/Red_ball/Both")
    parser.add_argument("--is_single_object", type=bool, default=False, help="Env contain only single object?")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes, default: 100")
    parser.add_argument("--max_ep_len", type=int, default=20, help="Number of timestep within an episode")
    parser.add_argument("--buffer_size", type=int, default=1200_00, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=50000, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--load_model", type=bool, default=False, help="Whether to load the model")
    parser.add_argument("--load_model_path", type=str, default="models/sac_agent", help="Model path directory")
    parser.add_argument("--fc_hidden_size", type=int, default=6656, help="Input size of the Lienar layer after cnv") #6656
    args = parser.parse_args()
    return args

def train(config):
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    if config.env ==  "SimplePickup":
        from env.env import MultiObjectMiniGridEnv
        env = MultiObjectMiniGridEnv(max_steps=config.max_ep_len, agent_view_size=5, size=7, \
                                     reward_object=config.env_reward_object, is_single_object = config.is_single_object)
        from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.load_model:
        model_dir = create_dump_directory(f"models/sac_agent/{config.env_reward_object}")
    else:
        model_dir = create_dump_directory("models/sac_agent")
    print("[INFO] Dump dir: ", model_dir)
    if config.env_reward_object != "Green_ball" and not config.load_model:
        warnings.warn("The rewarding object is not green ball and the model is not fine tuning")
        print("[INFO] Rewarding object: ", config.env_reward_object)
    
    steps = 0
    average = deque(maxlen=100)
    total_steps = 0
    
    with wandb.init(project="SAC_Image", name=config.env, config=config):
        sac_agent = SAC(input_dim = env.observation_space.shape[-1],
                        action_size = env.action_space.n,
                        fc_hidden_size = config.fc_hidden_size,
                        device=device,
                        buffer_size = config.buffer_size)

        if config.load_model:
            sac_agent.load_params(f"{config.load_model_path}/SAC_discrete.tar")
            print("[INFO] Fine tuning model is loaded from: ", config.load_model_path)
            print("[INFO] Single object is set to: ", config.is_single_object)

        sac_agent.to(device)
        # wandb.watch(sac_agent, log="gradients", log_freq=100)
        collect_random(env=env, dataset=sac_agent.buffer, load_model=config.load_model, num_samples=30000)
        total_steps += 30000
        for i in range(1, config.episodes+1):
            state, info = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action = sac_agent.get_action(np.transpose(state/255, (2,0,1)))
                steps += 1
                # p_state = env.get_unprocesed_obs()
                # p_agent_loc = env.agent_pos
                next_state, reward, terminated, truncated, _ = env.step(action)
                if not config.load_model:
                    class_prob = env.get_class(env.obs)
                else:
                    class_prob = 0
                done = terminated + truncated
                sac_agent.buffer.add(np.transpose(state/255, (2,0,1)), action, reward, np.transpose(next_state/255, (2,0,1)), done, class_prob)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = sac_agent.learn(sac_agent.buffer.sample(),
                                                                                                           gamma=0.99)
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break

            average.append(rewards)
            total_steps += episode_steps
            #print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
            wandb.log({"Reward": rewards/episode_steps,
                       "Average": np.mean(average),
                       "Episode steps" : episode_steps,
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Bellmann error 1": bellmann_error1,
                       "Bellmann error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Episode": i, 
                       "Buffer size": sac_agent.buffer.__len__()})


            if total_steps % config.save_every == 0:
                sac_agent.save(f"{model_dir}/", save_name="SAC_discrete")
        #Saving the model
        sac_agent.save(f"{model_dir}/", save_name="SAC_discrete")
        if not config.load_model:
            sac_agent.buffer.dump_data(f"data/{config.env}/SAC")

            
if __name__ == "__main__":
    config = get_config()
    train(config)
