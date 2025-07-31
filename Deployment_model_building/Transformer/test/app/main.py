import json
from flask import Flask, request, jsonify
from model_utils import load_model_and_tokenizer, predict_sentiment
from interpret import explain_prediction
# from app.model_utils import load_model_and_tokenizer, predict_sentiment
# from app.interpret import explain_prediction

# ✅ Load configuration from config.json
with open("config.json") as f:
# with open("config.json") as f:
    config = json.load(f)

HOST = config.get("host", "0.0.0.0")
PORT = config.get("port", 8002)
AUTH_TOKEN = config.get("auth_token")

ENDPOINT = config.get("transformer_endpoint")

app = Flask(__name__)
model, tokenizer, label_encoder = load_model_and_tokenizer()

@app.route(ENDPOINT, methods=['POST'])
def predict():
    data = request.get_json()

    # ✅ Authenticate using auth_token
    if data.get("auth_token") != AUTH_TOKEN:
        return jsonify({"error": "Unauthorized access"}), 401

    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = predict_sentiment(text, model, tokenizer, label_encoder)
    explanation = explain_prediction(text, model, tokenizer, label_encoder)
    return jsonify({
        'prediction': prediction,
        'interpretation': explanation
    })

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
