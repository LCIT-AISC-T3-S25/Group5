import torch
import json
import logging
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from generator import Generator
from utils import generate_image
from torchvision.utils import save_image

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load configuration from config.json
with open("config.json") as f:
    config = json.load(f)

HOST = config.get("host", "0.0.0.0")
PORT = config.get("port", 8000)
AUTH_TOKEN = config.get("auth_token")
ENDPOINT = config.get("dcgan_endpoint", "/generate")

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("dcgan_generator.pth", map_location=device))
generator.eval()

# Helper: convert image tensor to base64 string
def image_to_base64(image_tensor):
    buffer = BytesIO()
    save_image(image_tensor, buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})

# Image generation endpoint
@app.route(ENDPOINT, methods=["POST"])
def generate():
    try:
        # Authorization check
        auth = request.headers.get("Authorization")
        if auth != f"Bearer {AUTH_TOKEN}":
            return jsonify({"error": "Unauthorized"}), 401

        logging.info("Generating image...")
        image_tensor = generate_image(generator, device)
        img_base64 = image_to_base64(image_tensor)

        return jsonify({
            "message": "Image generated",
            "image_base64": img_base64
        })

    except Exception as e:
        logging.error(f"Generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run app using host/port from config
if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
