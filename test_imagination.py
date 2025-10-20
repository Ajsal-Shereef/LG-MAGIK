import os
import json
import hydra
import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from accelerate.utils import ProjectConfiguration
from architectures.common_utils import get_train_transform, save_gif, preprocess_llm_output, initialize_llm_hf_pipeline, query_llm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="config", config_name="test_imagination")
def main(args: DictConfig) -> None:
    if args.env.name ==  "SimplePickup":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.SimplePickup import SimplePickup
        env = SimplePickup(args.env, mode="Test")
        from minigrid.wrappers import RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env, tile_size=args.env.tile_size)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        env_name = env.unwrapped.env_name
        env_description = env.unwrapped.env_description
    if args.env.name ==  "Magik_env":
        if args.mode == "transfer":
            args.env.verbose = True
        args.env.is_single_object = False
        args.env.reward_object = "Both"
        from env.Magik_env import MultiObjectMiniGridEnv
        env = MultiObjectMiniGridEnv(args.env)
        from minigrid.wrappers import RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env, tile_size=args.env.tile_size)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        env_name = env.unwrapped.env_name
        env_description = env.unwrapped.env_description
    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    
    print("[INFO] Agent name: ", args.agent.name)
    print("[INFO] Env:", args.env.name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    #Make the agent
    args.agent.network.action_dim = int(env.action_space.n)
    args.agent.network.input_dim = int(env.observation_space.shape[-1])
    agent_model_dir = args.agent.evaluation.checkpoint
    if os.path.exists(agent_model_dir + "/config.yaml"):
        agent_model_args =  OmegaConf.load(agent_model_dir + "/config.yaml")
        args.agent = agent_model_args.agent
        args.agent.evaluation.checkpoint = agent_model_dir
    else:
        raise FileNotFoundError(f"Config file not found in {agent_model_dir}/config.yaml")
    agent = instantiate(args.agent.network)
    agent = agent.to(device)
    
    #Load the weights
    agent.load_params(args.agent.evaluation.checkpoint)
    
    # Set agent training params. Changing the epsilon to 0 to choose the acton greeedly
    args.agent.training.epsilon_start = 0
    agent.set_training_params(args.agent.training)
    
    # Make the vision model
    # Setup Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.models.accelerator.project_dir, 
        logging_dir=args.models.accelerator.logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.models.accelerator.gradient_accumulation_steps,
        mixed_precision=args.models.accelerator.mixed_precision,
        log_with=None, # Use the conditional logger
        project_config=accelerator_project_config,
    )
    if args.mode == "transfer":
        #Load the vision models
        vison_model_dir = args.models.test.model_dir
        if os.path.exists(vison_model_dir + "/config.yaml"):
            vision_model_args =  OmegaConf.load(vison_model_dir + "/config.yaml")
            cfg = vision_model_args.models
            args.models.test.model_dir = vison_model_dir
        else:
            raise FileNotFoundError(f"Config file not found in {vison_model_dir}/config.yaml")
        accelerator.print("Initializing VAE model...")
        vision_model = instantiate(cfg.model)
        vision_model.load_params(args.models.test.model_dir + f"/{cfg.project_name}.tar")

        # --- 5. Prepare agent for inference ---
        vision_model = accelerator.prepare(vision_model)

    # Get the mission
    mission = env.unwrapped.mission
    
    # Get data trasnformer
    train_transforms = get_train_transform()
    system_prompt = args.system_prompt
    # Load the .env file
    load_dotenv(dotenv_path="config/.env")

    if args.querry_mode == "openrouter":
        # Access the API key
        api_key = os.getenv('API_KEY')
        pipe = args.llm_model
    elif args.querry_mode == "huggingface":
        api_key = None
        pipe = initialize_llm_hf_pipeline(args.llm_model)
    elif args.querry_mode == "google":
        # Access the API key
        api_key = os.getenv('GOOGLE_API_KEY')
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        pipe = genai.GenerativeModel(args.llm_model)
        
    for episode in range(args.num_episode):
        frame_array_partial = []
        frame_array_full = []
        state, info = env.reset()
        episode_step = 1
        frame_array_full.append(env.unwrapped.get_frame())
        cumulative_reward = 0
        done = False
        while not done:
            if args.mode == "transfer":
                first_user_prompt = (
                                f"Environment description : {env_description}\n"
                                f"Target task : {mission}\n"
                                f"What agent knows : {args.agent.training.mission}.\n"
                                f"Input description: {info['description']}"
                            )
                if "No other objects can be seen" in info['description']:
                    llm_reply = info['description']
                else:
                    llm_reply, reasoning = query_llm(system_prompt, first_user_prompt, api_key, pipe, args.querry_mode)
                llm_reply_json = preprocess_llm_output(llm_reply)
                if llm_reply_json.get("imagine", False):
                    changed_state, imagined_state = vision_model.imagine(state, llm_reply_json.get("description", ""))
                else:
                    changed_state, imagined_state = train_transforms(state), state
            else:
                changed_state, imagined_state = train_transforms(state), state
            action = agent.get_action(changed_state, agent.initial_random_samples+1)
            next_state, reward, truncated, terminated, info = env.step(action)
            frame_array_partial.append(np.hstack((state, imagined_state)))
            frame_array_full.append(env.unwrapped.get_frame())
            done = truncated + terminated
            cumulative_reward += reward
            state = next_state
            episode_step += 1
        # write_video(frame_array, episode, dump_dir, frameSize=(env.unwrapped.get_frame().shape[1], env.unwrapped.get_frame().shape[0]))
        if args.mode == "transfer":
            save_dir = f"result/{args.agent.name}/{args.env.name}/{env_name}/transfer"
        else:
            save_dir = f"result/{args.agent.name}/{args.env.name}/{env_name}/source"
        save_gif(frame_array_partial, episode, save_dir, fps=args.env.fps, save_name= " partial")
        save_gif(frame_array_full, episode, save_dir, fps=args.env.fps, save_name= " full")
    
if __name__ == "__main__":
    main()