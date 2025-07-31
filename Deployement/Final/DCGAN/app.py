import torch
from flask import Flask, request, jsonify
import json
import logging
from generator import Generator
from utils import generate_image, save_image

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load configuration
with open("config.json") as f:
    config = json.load(f)

app = Flask(__name__)

# Load Generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("dcgan_generator.pth", map_location=device))
generator.eval()

# Health check endpoint (optional but useful for VM monitoring)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})

# Image generation endpoint
@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Authenticate request
        auth = request.headers.get("Authorization")
        if auth != f"Bearer {config['auth_token']}":
            return jsonify({"error": "Unauthorized"}), 401

        logging.info("Generating image...")
        image_tensor = generate_image(generator, device)
        img_path = save_image(image_tensor)
        logging.info(f"Image saved at: {img_path}")

        return jsonify({"image_path": img_path})

    except Exception as e:
        logging.error(f"Error during generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host=config["host"], port=config["port"])
