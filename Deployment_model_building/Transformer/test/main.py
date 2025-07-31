import json
from flask import Flask, request, jsonify
from model_utils import load_model_and_tokenizer, predict_sentiment
from interpret import explain_prediction

# ========== Load Configuration ==========
CONFIG_PATH = "config.json"

try:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"Configuration file '{CONFIG_PATH}' not found.")
except json.JSONDecodeError:
    raise RuntimeError(f"Invalid JSON in '{CONFIG_PATH}'.")

HOST = config.get("host", "0.0.0.0")
PORT = config.get("port", 8002)
AUTH_TOKEN = config.get("auth_token")
if not AUTH_TOKEN:
    raise RuntimeError("auth_token is missing in config.json")

# ========== Initialize App ==========
app = Flask(__name__)
model, tokenizer, label_encoder = load_model_and_tokenizer()

# ========== Routes ==========

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment with interpretation."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON or missing payload'}), 400

    if data.get("auth_token") != AUTH_TOKEN:
        return jsonify({"error": "Unauthorized access"}), 401

    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        prediction = predict_sentiment(text, model, tokenizer, label_encoder)
        explanation = explain_prediction(text, model, tokenizer, label_encoder)

        return jsonify({
            'prediction': prediction,
            'interpretation': explanation
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ========== Run Server ==========
if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
