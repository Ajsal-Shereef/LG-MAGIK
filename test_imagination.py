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
from architectures.common_utils import save_gif, preprocess_llm_output, initialize_llm_hf_pipeline, query_llm
from captioner import encode_image, query_llm as query_llm_vision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="config", config_name="test_imagination")
def main(args: DictConfig) -> None:
    if args.env.name ==  "SimplePickup":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.SimplePickup import SimplePickup
        env = SimplePickup(args.env)
        from minigrid.wrappers import RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env, tile_size=args.env.tile_size)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        env_name = env.unwrapped.env_name
        env_description = env.unwrapped.env_description
    elif args.env.name ==  "Magik_env":
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
    elif args.env.name ==  "PickEnv":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.PickEnv import PickEnv
        env = PickEnv(args.env, mode=args.env.mode)
        env_name = "PickEnv"
        env_description = env.env_description
    elif args.env.name == "MiniWorld":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.MiniWorld import PickObjectEnv
        env = PickObjectEnv(args.env)
        env_name = env.env_name
        env_description = env.env_description
    else:
        raise NotImplementedError("The environment is not implemented yet")
    
    
    print("[INFO] Agent name: ", args.agent_name)
    print("[INFO] Env:", args.env.name)
    print(f"[INFO] Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    #Make the agent
    if args.agent_name == "SAC":
        from stable_baselines3 import SAC
        agent = SAC.load(args.model_dir)
    elif args.agent_name == "PPO":
        from stable_baselines3 import PPO
        agent = PPO.load(args.model_dir)
    elif args.agent_name == "DQN":
        from stable_baselines3 import DQN
        agent = DQN.load(args.model_dir)
        
    agent_model_dir = args.model_dir
    if os.path.exists(os.path.dirname(agent_model_dir) + "/config.yaml"):
        agent_model_args =  OmegaConf.load(os.path.dirname(agent_model_dir) + "/config.yaml")
        args.env = agent_model_args.env
    else:
        raise FileNotFoundError(f"Config file not found in {os.path.dirname(agent_model_dir)}/config.yaml")
    
    # Get data trasnformer
    if args.env.get("observation_mode", "image") == "image":
        from architectures.common_utils import get_train_transform_cnn
        train_transforms = get_train_transform_cnn() 
    else:
        from architectures.common_utils import get_train_transform_mlp
        train_transforms = get_train_transform_mlp()  
    
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
        vision_model_path = args.models.test.model_dir
        vison_model_dir = os.path.dirname(args.models.test.model_dir)
        if os.path.exists(vison_model_dir + "/config.yaml"):
            vision_model_args =  OmegaConf.load(vison_model_dir + "/config.yaml")
            cfg = vision_model_args.models
            args.models.test.model_dir = vision_model_path
        else:
            raise FileNotFoundError(f"Config file not found in {vison_model_dir}/config.yaml")
        accelerator.print("Initializing VAE model...")
        vision_model = instantiate(cfg.model)
        vision_model.load_params(vision_model_path)

        # --- 5. Prepare agent for inference ---
        vision_model = accelerator.prepare(vision_model)
        
        #Setting the VAE model to eval mode
        vision_model.eval()
        
        system_prompt = args.system_prompt
        # Load the .env file
        load_dotenv(dotenv_path="config/.env")
    
        if args.querry_mode == "openrouter":
            # Access the API key
            api_key = os.getenv('OPEN_ROUTER_API_KEY')
            pipe = args.llm_model
            alternative_pipe = args.alternate_llm_model
        elif args.querry_mode == "huggingface":
            api_key = None
            pipe = initialize_llm_hf_pipeline(args.llm_model)
            alternative_pipe = None
        elif args.querry_mode == "google":
            # Access the API key
            api_key = os.getenv('GOOGLE_API_KEY')
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            pipe = genai.GenerativeModel(args.llm_model)
            alternative_pipe = None

    # Get the mission
    mission = env.unwrapped.mission
  
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
                                f"What agent knows : {args.env.mission}.\n"
                                f"Input description: {info['description']}"
                            )
                if args.get("llm_caption", False):
                    # Capture the frame
                    frame = state
                    # Encode the frame
                    base64_image = encode_image(frame)
                    
                    sensor_data = env.unwrapped.get_sensor_data()
                    prompt_text = args.vision_prompt_text.format(sensor_data=sensor_data)
                    vision_prompt = [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                    
                    try:
                        caption = query_llm_vision(
                            system=args.caption_system_prompt,
                            prompt=vision_prompt,
                            api_key=api_key,
                            mode=args.querry_mode,
                            pipeline=args.vllm_model,
                            temperature=0.1
                        )
                        info['description'] = caption
                        # Reconstruct user prompt with new description
                        first_user_prompt = (
                                f"Environment description : {env_description}\n"
                                f"Target task : {mission}\n"
                                f"What agent knows : {args.env.mission}.\n"
                                f"Input description: {info['description']}"
                            )
                        # print(f"[INFO] Updated description via LLM: {caption}")
                    except Exception as e:
                        print(f"[ERROR] LLM Captioning failed: {e}")

                if "No other objects can be seen." in info['description']:
                    llm_reply = info['description']
                else:
                    llm_reply, reasoning = query_llm(system_prompt, first_user_prompt, api_key, pipe, alternative_pipe, args.querry_mode)
                llm_reply_json = preprocess_llm_output(llm_reply)
                if llm_reply_json.get("imagine", False):
                    changed_state, imagined_state = vision_model.imagine(state, llm_reply_json.get("description", ""))
                else:
                    changed_state, imagined_state = train_transforms(state), state
            else:
                changed_state, imagined_state = train_transforms(state), state
            action = agent.predict(imagined_state, deterministic=False)[0]
            next_state, reward, truncated, terminated, info = env.step(action)
            frame_array_partial.append(np.hstack((state, imagined_state)))
            frame_array_full.append(env.unwrapped.get_frame())
            done = truncated + terminated
            cumulative_reward += reward
            state = next_state
            print(f"Episode step done: {episode_step}")
            episode_step += 1
        # write_video(frame_array, episode, dump_dir, frameSize=(env.unwrapped.get_frame().shape[1], env.unwrapped.get_frame().shape[0]))
        if args.mode == "transfer":
            save_dir = f"result/{args.agent_name}/{args.env.name}/{env_name}/transfer"
        else:
            save_dir = f"result/{args.agent_name}/{args.env.name}/{env_name}/source"
        save_gif(frame_array_partial, episode, save_dir, fps=args.env.fps, save_name= " partial")
        save_gif(frame_array_full, episode, save_dir, fps=args.env.fps, save_name= " full")
        print(f"----------- Episode done:  {episode} ----------------")
    
    if args.env.name ==  "SimplePickup":
        agent_performance = env.unwrapped.get_performance_metric()
    else:
        agent_performance = env.get_performance_metric()
    print("Agent performance" , agent_performance)
    
if __name__ == "__main__":
    main()