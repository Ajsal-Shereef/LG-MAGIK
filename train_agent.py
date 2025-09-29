import torch
import wandb
import hydra
import warnings
import numpy as np

from PIL import Image
from collections import deque
from architectures.common_utils import *
from omegaconf import DictConfig, OmegaConf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="config", config_name="train_agent")
def train(args: DictConfig) -> None:
    # np.random.seed(config.seed)
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    
    if args.env.name ==  "SimplePickup":
        from env.SimplePickup import SimplePickup
        env = SimplePickup(args.env, mode="Test")
        from minigrid.wrappers import RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env, tile_size=args.env.tile_size)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    if args.use_wandb:
        wandb.init(project="LG-MAGIK", name=f"{args.agent.name}_{args.env.name}", config=OmegaConf.to_container(args, resolve=True))

    print("[INFO] Agent name: ", args.agent.name)
    print("[INFO] Env:", args.env.name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    #Make the agent
    args.agent.network.action_dim = int(env.action_space.n)
    args.agent.network.input_dim = int(env.observation_space.shape[-1])
    agent = hydra.utils.instantiate(args.agent.network)
    agent = agent.to(device)
    
    # Initilising buffer
    agent.initialise_buffer(args.agent.hyperparameters)
    
    # Set training params
    agent.set_training_params(args.agent.training)
    
    # Initialise the optimizer
    agent.set_optimizer(args.agent.hyperparameters)
    
    model_dir = create_dump_directory(f"model_weights/{args.agent.name}")
    print("[INFO] Dump dir: ", model_dir)
    
    # Dumping the training config files
    config_path = os.path.join(model_dir, "config.yaml")
    OmegaConf.save(config=args, f=config_path)
    
    train_transforms = get_train_transform()
    
    env_total_steps = 0
    env_episode_steps = 0
    env_episodes = 0
    agent.do_pre_task_proceessing()   
    state, info = env.reset()
    cumulative_reward = 0
    average_episodic_return = deque(maxlen=10)
    for i in range(1, args.env.total_timestep+1):
        action = agent.get_action(train_transforms(state), env_total_steps)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.add_transition_to_buffer((state, action, reward, next_state, terminated, truncated))
        metric = agent.learn(env_total_steps)
        state = next_state
        cumulative_reward += reward
        env_total_steps += 1
        env_episode_steps += 1
        metric["Returns"] = cumulative_reward
        metric["Average episodic returns"] = np.mean(average_episodic_return) if len(average_episodic_return) > 0 else 0
        metric["Episode steps"] = env_episode_steps
        metric["Env total steps"] = env_total_steps
        metric["Env episode"] = env_episodes
        metric["Buffer size"] = agent.buffer.__len__()
        if done:
            state, info = env.reset()
            env_episodes += 1
            average_episodic_return.append(cumulative_reward)
            env_episode_steps = 0
            cumulative_reward = 0
            agent.do_post_episode_processing(env_total_steps)
            
        if args.use_wandb and env_total_steps%args.log_every==0:
            wandb.log(metric)
        if i % args.save_every == 0:
            agent.save(f"{model_dir}/", save_name=f"{args.agent.name}")
    agent.eval()
    dump_dir = args.agent.video_save_path + f"/{args.agent.name}"
    agent.test(env, args.env.fps, dump_dir)
    #Saving the model
    agent.save(f"{model_dir}/", save_name=f"{args.agent.name}")
    
if __name__ == "__main__":
    train()