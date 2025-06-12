from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import pickle
from waitress import serve

# Load model and supporting files
try:
    model = tf.keras.models.load_model("Bidirectional_GRU_SavedModel", compile=False)  # ✅ .h5 format

    # Load tokenizer from JSON string
    with open("tokenizer.json") as f:
        tokenizer = tokenizer_from_json(f.read())

# Load label encoder from list
    with open("gru_encoder.json") as f:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(json.load(f))
    MAX_LEN = 100
except Exception as e:
    print(f"Error loading model or files: {str(e)}")
    raise

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model": "Bidirectional GRU Sentiment Analysis"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)

        prediction = model.serve(padded).numpy()
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

        return jsonify({
            "input_text": text,
            "predicted_sentiment": predicted_label[0],
            "confidence": float(np.max(prediction)),
            "model": "Bidirectional GRU"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
