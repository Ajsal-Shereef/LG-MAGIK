import sys
import os
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from architectures.common_utils import get_train_transform_cnn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = None

@app.on_event("startup")
async def startup_event():
    global vision_model
    try:
        # Initialize hydra with config path relative to this file
        # ../../config maps to project_root/config
        with initialize(version_base=None, config_path="../../config"):
            args = compose(config_name="test_imagination")
            
        print(f"[INFO] Loading model from: {args.models.test.model_dir}")
        
        vision_model_path = args.models.test.model_dir
        model_dir = os.path.dirname(vision_model_path)
        model_config_path = os.path.join(model_dir, "config.yaml")
        
        if os.path.exists(model_config_path):
             print(f"[INFO] Loading specific model config from {model_config_path}")
             vision_model_args = OmegaConf.load(model_config_path)
             cfg = vision_model_args.models
        else:
             print("[INFO] Using default config")
             cfg = args.models
             
        vision_model = instantiate(cfg.model)
        vision_model.load_params(vision_model_path)
        vision_model.to(device)
        vision_model.eval()
        
        # Ensure we have the transform (imagine function uses self.train_transform, 
        # but the class requires it to be set or it initializes it? 
        # In TextConditionedVAE.__init__, it calls get_train_transform_cnn(). So it is self-contained.)
        
        print("[INFO] Model loaded successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()

@app.post("/imagine")
async def imagine(file: UploadFile = File(...), caption: str = Form(...)):
    global vision_model
    if not vision_model:
        raise HTTPException(status_code=500, detail="Model not loaded or failed to initialize")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # The imagine method returns (tensor, numpy_uint8)
        _, imagined_numpy = vision_model.imagine(image_np, caption)
        
        res_image = Image.fromarray(imagined_numpy)
        img_byte_arr = io.BytesIO()
        res_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    with open(os.path.join(os.path.dirname(__file__), "../frontend/index.html"), "r") as f:
        html_content = f.read()
    return Response(content=html_content, media_type="text/html")
