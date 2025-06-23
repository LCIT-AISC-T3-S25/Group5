import os
import numpy as np
import pickle
from fastapi import APIRouter, UploadFile, File, Form
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from io import BytesIO
import json

router = APIRouter()

# -------------------------------
# Asset paths
# -------------------------------
MODEL_PATH = os.path.join("model", "vgg16_model.h5")
TOKENIZER_PATH = os.path.join("model", "tokenizer.json")
ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

# -------------------------------
# Load model and assets
# -------------------------------
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "r") as f:
    tokenizer = tokenizer_from_json(f.read())

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------------
# Prediction Endpoint
# -------------------------------
@router.post("/predict")
async def predict(file: UploadFile = File(...), categories: str = Form(...)):
    try:
        # Image preprocessing
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # Metadata preprocessing
        sequence = tokenizer.texts_to_sequences([categories])
        if not sequence or len(sequence[0]) == 0:
            return {"error": "Tokenizer produced empty sequence. Check category format."}

        sequence = pad_sequences(sequence, maxlen=500)  # ✅ Match your model's expected input shape

        # Make prediction
        prediction = model.predict([image_array, sequence])
        label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        return {
            "predicted_label": label,
            "confidence": float(np.max(prediction))
        }

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"error": str(e)}