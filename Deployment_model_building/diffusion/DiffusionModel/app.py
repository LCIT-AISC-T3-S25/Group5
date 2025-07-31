import os
import torch
from flask import Flask, request, jsonify
from utils import DiffusionModel, denormalize
import clip
from PIL import Image
import json
import uuid
import base64
from io import BytesIO

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

IMG_SIZE = config["img_size"]
TIMESTEPS = config["timesteps"]
EMBEDDING_DIM = config["embedding_dim"]
LABELS = config["labels"]
HOST = config.get("host", "0.0.0.0")
PORT = config.get("port", 8001)
ENDPOINT = config.get("diffusion_endpoint", "/generate")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _ = clip.load("ViT-B/32", device=device)
model = DiffusionModel(clip_model, embedding_dim=EMBEDDING_DIM).to(device)

# Load model weights
model_path = "diffusion_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Loaded diffusion model.")
else:
    print("❌ Model weights not found!")

# Diffusion constants
beta = torch.linspace(0.0001, 0.02, TIMESTEPS).to(device)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def tokenize(captions):
    return clip.tokenize(captions).to(device)

def generate_from_caption(caption):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        tokens = tokenize([caption])

        for t in range(TIMESTEPS - 1, -1, -1):
            t_tensor = torch.full((1,), t, device=device).long()
            noise_pred = model(x, t_tensor, tokens)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha[t])) * (
                x - ((1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t])) * noise_pred
            ) + torch.sqrt(beta[t]) * noise

        img = denormalize(x[0].cpu()).permute(1, 2, 0).numpy()
        img = (img * 255).astype("uint8")
        return Image.fromarray(img)

app = Flask(__name__)

@app.route(ENDPOINT, methods=["POST"])
def generate():
    data = request.get_json()
    caption = data.get("prompt", "")
    if not caption:
        return jsonify({"error": "Prompt missing"}), 400

    image = generate_from_caption(caption)

    # Convert image to base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({
        "image_base64": base64_image,
        "message": "Image generated"
    })

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
