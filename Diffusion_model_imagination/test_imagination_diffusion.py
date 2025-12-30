
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

# Import necessary utils from original codebase
from architectures.common_utils import save_gif, preprocess_llm_output, initialize_llm_hf_pipeline, query_llm
from captioner import encode_image, query_llm as query_llm_vision

# Import our new Diffusion Model
from Diffusion_model_imagination.models.diffusion import DiffusionImaginationModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="../config", config_name="test_imagination") # Reusing test_imagination config base
def main(args: DictConfig) -> None:
    # --- 1. Environment Setup (Same as original) ---
    if args.env.name ==  "SimplePickup":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.SimplePickup import SimplePickup
        env = SimplePickup(args.env)
        from minigrid.wrappers import RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env, tile_size=args.env.tile_size)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
    elif args.env.name ==  "Magik_env":
         # ... copy existing environment setup logic ...
         # For brevity, assuming user uses MiniWorld or Magik as per original script. 
         # I will replicate the logic for MiniWorld as it's common.
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
    elif args.env.name == "MiniWorld":
        if args.mode == "transfer":
            args.env.verbose = True
        from env.MiniWorld import PickObjectEnv
        env = PickObjectEnv(args.env)
    else:
        # Fallback or strict requirement
        # raise NotImplementedError("Env not fully supported in this snippet, add others as needed")
        pass
    
    env_name = getattr(env, "env_name", args.env.name)
    env_description = getattr(env, "env_description", "")

    # --- 2. Agent Setup (Same as original) ---
    if args.agent_name == "SAC":
        from stable_baselines3 import SAC
        agent = SAC.load(args.model_dir)
    elif args.agent_name == "PPO":
        from stable_baselines3 import PPO
        agent = PPO.load(args.model_dir)
    elif args.agent_name == "DQN":
        from stable_baselines3 import DQN
        agent = DQN.load(args.model_dir)
        
    # Data transforms
    if args.env.get("observation_mode", "image") == "image":
        from architectures.common_utils import get_train_transform_cnn
        train_transforms = get_train_transform_cnn() 
    else:
        from architectures.common_utils import get_train_transform_mlp
        train_transforms = get_train_transform_mlp()

    # --- 3. Initialize Diffusion Model for Imagination ---
    accelerator = Accelerator(mixed_precision=args.models.accelerator.mixed_precision)
    
    if args.mode == "transfer":
        # Load VAE first
        vision_model_path = args.models.test.model_dir # This points to VAE path usually
        vae_dir = os.path.dirname(vision_model_path)
        if os.path.exists(vae_dir + "/config.yaml"):
            vae_args = OmegaConf.load(vae_dir + "/config.yaml")
            vae_cfg = vae_args.models.model
        
        accelerator.print(f"Loading VAE from {vision_model_path}")
        vae = instantiate(vae_cfg)
        vae.load_params(vision_model_path)
        vae.eval()
        
        # Initialize Diffusion wrapper
        # Ideally we load a trained diffusion checkpoint here. 
        # For now, initializing from scratch or user provided path if created.
        diffusion_model = DiffusionImaginationModel(vae)
        
        # Load Diffusion Weights if available
        # diffusion_weight_path = args.get("diffusion_weight_path", None)
        # if diffusion_weight_path:
        #      diffusion_model.unet.from_pretrained(diffusion_weight_path)
        
        diffusion_model = accelerator.prepare(diffusion_model)
        diffusion_model.eval()

    # --- 4. LLM Setup (Same as original) ---
    load_dotenv(dotenv_path="config/.env")
    # ... (LLM setup code omitted for brevity but should be included) ...
    # Assuming valid 'pipe' and 'query_llm' function
    api_key = os.getenv('OPEN_ROUTER_API_KEY') # Example
    # For this snippet, assuming standard LLM pipeline setup is done or passed via args

    # --- 5. Main Loop ---
    # Simplified loop
    mission = env.unwrapped.mission
    
    for episode in range(args.num_episode):
        state, info = env.reset()
        done = False
        
        while not done:
            if args.mode == "transfer":
                # ... Description Logic ...
                # Use LLM to get 'llm_reply_json' with 'description' (Source Aligned)
                # Hardcoding or assuming logic exists.
                
                # Placeholder for LLM logic
                llm_reply_json = {"imagine": True, "description": "A blue box"} # Example
                
                if llm_reply_json.get("imagine", False):
                    # --- DIFFUSION IMAGINATION ---
                    # 1. Transform State
                    if args.env.get("observation_mode", "image") == "image":
                       # Convert to tensor [1, C, H, W]
                       state_tensor = train_transforms(Image.fromarray(state)).unsqueeze(0).to(device)
                    
                    # 2. Get Text Embeddings
                    # We need to tokenize the description
                    # Using VAE's tokenizer/encoder
                    target_desc = llm_reply_json["description"]
                    tokenizer = vae.decoder.tokenizer
                    text_inputs = tokenizer(
                        target_desc, 
                        max_length=tokenizer.model_max_length, 
                        padding="max_length", 
                        truncation=True, 
                        return_tensors="pt"
                    )
                    with torch.no_grad():
                         text_embeddings = vae.decoder.text_encoder(
                             text_inputs.input_ids.to(device), 
                             attention_mask=text_inputs.attention_mask.to(device)
                         )[0]
                    
                    # 3. Imagine
                    with torch.no_grad():
                        imagined_latent = diffusion_model.imagine(
                            state_tensor, 
                            text_embeddings, 
                            strength=0.6, # Tuning parameter
                            num_inference_steps=20 
                        )
                        # Decode
                        # Assuming VAE decoder reconstructs from latent
                        # We might need to pass dummy text if VAE decoder requires it
                        # But ideally we want purely visual decode. 
                        # TextConditionedDecoder uses text. We pass the SOURCE description (Blue Box)
                        imagined_image, _ = vae.decoder(
                            imagined_latent, 
                            text_inputs.input_ids.to(device), 
                            text_inputs.attention_mask.to(device),
                            return_text_feats=False
                        )
                        
                    # Convert back to numpy for Agent
                    imagined_state = ((imagined_image.squeeze().detach().cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0) * 255).astype(np.uint8)
                    
                else:
                     imagined_state = state
                     
                action = agent.predict(imagined_state, deterministic=False)[0]
            else:
                action = agent.predict(state, deterministic=False)[0]
                
            state, reward, truncated, terminated, info = env.step(action)
            done = truncated + terminated

if __name__ == "__main__":
    main()
