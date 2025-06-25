from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "proxy": True})

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content, skips 404

# Proxy for text models (GRU and Word2Vec)
@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.get_json()
    text = data.get("text", "")
    model = data.get("model", "")

    if not text:
        return jsonify({"error": "Text input missing"}), 400

    port_map = {
        "gru": "5004",
        "word2vec": "5003"
    }

    if model not in port_map:
        return jsonify({"error": "Invalid model selected"}), 400

    backend_url = f"http://host.docker.internal:{port_map[model]}/predict"

    try:
        response = requests.post(backend_url, json={"text": text})
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error contacting model backend: {str(e)}"}), 500

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image_file = request.files["file"]
    model = request.form.get("model", "")

    if model not in ["vgg16", "cnn"]:
        return jsonify({"error": "Invalid image model selected"}), 400

    port_map = {
        "vgg16": "5005",
        "cnn": "5006"
    }

    backend_url = f"http://host.docker.internal:{port_map[model]}/predict"

    file_bytes = image_file.read()

    files = {
        "file": (image_file.filename, file_bytes, image_file.mimetype)
    }

    data = {}
    # Only add categories if model is vgg16
    if model == "vgg16":
        categories = request.form.get("categories", "")
        data["categories"] = categories

    try:
        response = requests.post(backend_url, files=files, data=data)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error forwarding to image model: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
