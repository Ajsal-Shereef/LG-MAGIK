import sys
import os
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from hydra import compose, initialize
from omegaconf import OmegaConf
from hydra.utils import instantiate
import base64
import uvicorn

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
        
        print("[INFO] Model loaded successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()

def encode_image_base64(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return "data:image/png;base64," + base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

def create_latent_grid(mean_tensor, ref_tensor=None):
    # Normalize each channel for visualization
    # mean_tensor: [1, latent_channels, H, W]
    mean = mean_tensor.squeeze(0) # [latent_channels, H, W]
    
    if ref_tensor is not None:
        ref = ref_tensor.squeeze(0)
    else:
        ref = mean
        
    num_channels = mean.shape[0]
    
    # Create a grid with padding
    cols = 4
    rows = (num_channels + cols - 1) // cols
    
    channels_np = mean.cpu().numpy()
    ref_np = ref.cpu().numpy()
    
    h, w = channels_np.shape[1], channels_np.shape[2]
    padding = 1
    
    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    
    grid_img = Image.new('L', (grid_w, grid_h), color=255)
    
    for i in range(num_channels):
        ch_data = channels_np[i]
        
        # Use REFERENCE min/max for normalization to preserve relative changes
        ch_min = ref_np[i].min()
        ch_max = ref_np[i].max()
        
        # Avoid division by zero
        if ch_max - ch_min > 1e-5:
            # We must clip because modified vals might exceed orig range
            ch_norm = (ch_data - ch_min) / (ch_max - ch_min) * 255
            ch_norm = np.clip(ch_norm, 0, 255)
        else:
            ch_norm = np.zeros_like(ch_data)
        
        ch_img = Image.fromarray(ch_norm.astype(np.uint8))
        
        r = i // cols
        c = i % cols
        
        x_pos = padding + c * (w + padding)
        y_pos = padding + r * (h + padding)
        
        grid_img.paste(ch_img, (x_pos, y_pos))
        
    if grid_img.width < 512:
        scale = 512 / grid_img.width
        new_size = (int(grid_img.width * scale), int(grid_img.height * scale))
        grid_img = grid_img.resize(new_size, Image.NEAREST)
    return grid_img

@app.post("/imagine")
async def imagine(
    file: UploadFile = File(...), 
    caption: str = Form(...),
    mode: str = Form("imagination"), # Default to imagination
    channel_scales: str = Form(None) # JSON string of scales
):
    global vision_model
    if not vision_model:
        raise HTTPException(status_code=500, detail="Model not loaded or failed to initialize")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        response_data = {}
        
        if mode == "latent":
            # --- Latent Visualization & Manipulation Logic ---
            transform = vision_model.train_transform
            state_tensor = transform(image_np if isinstance(image_np, np.ndarray) else np.array(image)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Encode
                hidden = vision_model.encoder(state_tensor)
                sampler = vision_model.bottleneck(hidden)
                mean = sampler.mean # [1, latent_channels, H, W]
                
                # Check if we have scaling factors (manipulation mode)
                if channel_scales:
                    import json
                    scales = json.loads(channel_scales)
                    
                    # with open("server_log.txt", "a") as logf:
                    #     logf.write(f"Received scales: {scales}\n")
                    #     logf.write(f"Mean stats before: Min={mean.min().item()}, Max={mean.max().item()}, Avg={mean.mean().item()}\n")
                    
                    # scales should be list of length latent_channels
                    if len(scales) == mean.shape[1]:
                        # Apply scaling per channel
                        modified_mean = mean.clone()
                        for i, s in enumerate(scales):
                            modified_mean[:, i, :, :] *= float(s)
                            
                        with open("server_log.txt", "a") as logf:
                            logf.write(f"Mean stats after scaling: Min={modified_mean.min().item()}, Max={modified_mean.max().item()}, Avg={modified_mean.mean().item()}\n")
                            
                        # Decode to get reconstruction
                        if hasattr(vision_model, "decoder") and hasattr(vision_model.decoder, "tokenizer"):
                             tokeniser = vision_model.decoder.tokenizer
                             # Import tokenize_captions function
                             from architectures.common_utils import tokenize_captions
                             # Use the utility function directly
                             captions_tokenised, attention_mask = tokenize_captions(tokeniser, [caption if caption else ""], max_length=vision_model.max_sequence_length)
                             captions_tokenised = captions_tokenised.to(device)
                             attention_mask = attention_mask.to(device)
                             
                             reconstructed_x, _ = vision_model.decoder(modified_mean, captions_tokenised, attention_mask, return_text_feats=True)
                             
                        else:
                             reconstructed_x = vision_model.decode(modified_mean).sample
                             
                        # Post-process reconstruction
                        imagined_numpy = ((reconstructed_x.squeeze().detach().cpu().numpy() * 0.5 + 0.5).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                        res_image = Image.fromarray(imagined_numpy)
                        response_data["reconstruction"] = encode_image_base64(res_image)
                        
                        # Generate "Modified Latent" Grid
                        # We use the modified_mean DIRECTLY to show exactly what was fed to the decoder.
                        # This ensures that if only Ch-i was changed, only Ch-i changes in the vis.
                        mod_grid = create_latent_grid(modified_mean, ref_tensor=mean)
                        response_data["modified_latent"] = encode_image_base64(mod_grid)
                        
                    else:
                        raise ValueError(f"Scales length {len(scales)} does not match latent channels {mean.shape[1]}")
                        
                else:
                    # Default: Initial Latent Viz (No scales provided)
                    # We return the "Original Latent" grid
                    grid_img = create_latent_grid(mean)
                    response_data["original_latent"] = encode_image_base64(grid_img)
            
        else:
            # --- Standard Imagination Logic ---
            _, imagined_numpy = vision_model.imagine(image_np, caption)
            res_image = Image.fromarray(imagined_numpy)
            response_data["result"] = encode_image_base64(res_image)
            
            # --- Latent Visualization for Imagination Mode ---
            with torch.no_grad():
                # 1. Original Image Latent
                transform = vision_model.train_transform
                state_tensor = transform(image_np if isinstance(image_np, np.ndarray) else np.array(image)).unsqueeze(0).to(device)
                hidden = vision_model.encoder(state_tensor)
                sampler = vision_model.bottleneck(hidden)
                mean_original = sampler.mean
                
                # 2. Reconstructed Image Latent
                # We need to process the imagined_numpy back to tensor to get its latent
                # imagined_numpy is uint8 [H, W, 3]
                imagined_pil = Image.fromarray(imagined_numpy).convert("RGB")
                imagined_tensor = transform(imagined_pil).unsqueeze(0).to(device)
                hidden_recon = vision_model.encoder(imagined_tensor)
                sampler_recon = vision_model.bottleneck(hidden_recon)
                mean_recon = sampler_recon.mean
                
                # 3. Generate Grids
                # Use mean_original as the reference for normalization for both to allow comparison
                grid_original = create_latent_grid(mean_original)
                grid_recon = create_latent_grid(mean_recon, ref_tensor=mean_original)
                
                response_data["original_latent"] = encode_image_base64(grid_original)
                response_data["reconstructed_latent"] = encode_image_base64(grid_recon)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    with open(os.path.join(os.path.dirname(__file__), "../frontend/index.html"), "r") as f:
        html_content = f.read()
    return Response(content=html_content, media_type="text/html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
