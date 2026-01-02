# Diffusion Imagination Web App

This is a web application for testing the pixel-space diffusion model with interactive controls.

## Prerequisites

Ensure you have the project dependencies installed. The app requires `fastapi`, `uvicorn`, `torch`, `diffusers`, `transformers`, and `hydra-core`.

## How to Run

1.  Navigate to the root directory of the repository (e.g., `LG-MAGIK`).
2.  Run the following command to start the server:

    ```bash
    uvicorn Diffusion_model_imagination.web_app.app:app --host 0.0.0.0 --port 8000
    ```

    *Note: The first run might take a minute to load the model weights.*

3.  Open your web browser and navigate to:
    [http://localhost:8000](http://localhost:8000)

## Features

- **Source Image**: Upload an initial image to guide the generation.
- **Prompt**: Describe the scene or changes you want.
- **Strength**: Control how much the generated image deviates from the source (0.0 = original, 1.0 = full noise).
- **Guidance Scale**: Control how strongly the image adheres to your text prompt.
