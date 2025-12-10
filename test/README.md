# VAE Imagination App

This directory contains a web application to demonstrate the Text Conditioned VAE.

## Structure
- `backend/`: FastAPI backend server.
- `frontend/`: HTML frontend interface.

## How to Run

**Important**: You must run the backend from the **root directory** of the project (`LG-MAGIK/`) to ensure all imports work correctly.

### 1. Install Dependencies
```bash
pip install fastapi uvicorn python-multipart hydra-core omegaconf torch torchvision pillow
```

### 2. Start Backend
From the project root (`LG-MAGIK/`):
```bash
python3 -m uvicorn test.backend.app:app --host 0.0.0.0 --port 8000
```
Wait until you see `[INFO] Model loaded successfully`.

### 3. Open Frontend
Open `http://localhost:8000/` in your web browser.
(Ensure you have port forwarding set up for port 8000 if running remotely)
