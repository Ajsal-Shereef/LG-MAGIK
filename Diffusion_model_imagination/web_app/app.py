
import sys
import os
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, FileResponse
from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.utils import instantiate
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from architectures.common_utils import get_train_transform_cnn
# Import our new Diffusion Model
from Diffusion_model_imagination.models.diffusion import DiffusionImaginationModel


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global diffusion_model, tokenizer, text_encoder, train_transforms
    try:
        # Initialize hydra with config path
        # ../../config maps to project_root/config
        with initialize(version_base=None, config_path="../../config"):
            args = compose(config_name="train_diffusion") # Using train_diffusion config as base for paths
            
        print(f"[INFO] Config Loaded. Device: {device}")
        
        # 1. Load CLIP Text Encoder & Tokenizer
        text_encoder_path = args.models.data.text_encoder_path
        print(f"[INFO] Loading CLIP Text Encoder from: {text_encoder_path}")
        try:
             # Try loading local first if it's a path
             if os.path.exists(text_encoder_path):
                 tokenizer = CLIPTokenizer.from_pretrained(text_encoder_path)
                 text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
             else:
                 # Fallback to huggingface hub
                 tokenizer = CLIPTokenizer.from_pretrained(text_encoder_path)
                 text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        except Exception as e:
            print(f"[WARNING] Failed to load CLIP from {text_encoder_path}: {e}")
            print("[INFO] Fallback to 'openai/clip-vit-base-patch32'")
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        text_encoder.to(device)
        text_encoder.eval()
        
        # 2. Load Diffusion Model
        print("[INFO] Initializing Diffusion Model...")
        
        # Instantiate the model wrapper
        diffusion_model_wrapper = DiffusionImaginationModel(unet_config=None) 
        
        # Load weights if available in config
        if "test" in args.models and "model_dir" in args.models.test:
             model_path = args.models.test.model_dir
             print(f"[INFO] Loading Diffusion weights from: {model_path}")
             try:
                 # Attempt to load U-Net weights
                 # Check if it's a directory with unet config
                 diffusion_model_wrapper.unet = UNet2DConditionModel.from_pretrained(model_path)
             except Exception as e:
                 print(f"[ERROR] convert via from_pretrained failed: {e}")
                 # Fail hard if we expect a model but can't load it
                 raise RuntimeError(f"Could not load model from {model_path}. Error: {e}")
                      
             diffusion_model_wrapper.to(device)
             diffusion_model_wrapper.eval()
             diffusion_model = diffusion_model_wrapper
             
        else:
            print("[WARNING] No test model_dir found in config. Using random initialization.")
            diffusion_model_wrapper.to(device)
            diffusion_model_wrapper.eval()
            diffusion_model = diffusion_model_wrapper
        
        # 3. Transforms
        train_transforms = get_train_transform_cnn()
        
        print("[INFO] Models loaded successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        import traceback
        traceback.print_exc()

    yield
    # Clean up resources if needed
    print("[INFO] Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion_model = None
tokenizer = None
text_encoder = None
train_transforms = None

@app.post("/generate")
async def generate(
    file: UploadFile = File(...), 
    prompt: str = Form(...),
    strength: float = Form(0.6),
    guidance_scale: float = Form(7.5),
    mode: str = Form("sdedit")
):
    global diffusion_model, tokenizer, text_encoder, train_transforms
    
    if not diffusion_model or not text_encoder:
        raise HTTPException(status_code=500, detail="Models not loaded")
        
    try:
        # 1. Process Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((80, 80)) 
        state_tensor = train_transforms(image).unsqueeze(0).to(device)
        
        # 2. Process Text
        # Target Prompt
        text_inputs = tokenizer(
            prompt, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
             text_embeddings = text_encoder(
                 text_inputs.input_ids.to(device), 
                 attention_mask=text_inputs.attention_mask.to(device)
             )[0]
             
        # Null Prompt (Unconditional)
        null_inputs = tokenizer(
            [""], 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
             null_embeddings = text_encoder(
                 null_inputs.input_ids.to(device), 
                 attention_mask=null_inputs.attention_mask.to(device)
             )[0]
        
        # 3. Generate (SDEdit only)
        with torch.no_grad():
            imagined_tensor = diffusion_model.imagine(
                state_image=state_tensor, 
                target_text_embeddings=text_embeddings,
                uncond_text_embeddings=null_embeddings, 
                strength=strength, 
                num_inference_steps=50, 
                guidance_scale=guidance_scale
            )
            
        # 4. Post-process
        # Tensor [-1, 1] -> Numpy [0, 255]
        imagined_numpy = ((imagined_tensor.squeeze().detach().cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0) * 255).astype(np.uint8)
        
        res_image = Image.fromarray(imagined_numpy)
        img_byte_arr = io.BytesIO()
        res_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static/index.html"))
