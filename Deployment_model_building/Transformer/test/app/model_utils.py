import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from positional_encoding import PositionalEncoding, TransformerBlock

# ✅ Load configuration
with open("config.json") as f:
    CONFIG = json.load(f)

MAX_LEN = CONFIG.get("max_len", 200)
MODEL_PATH = CONFIG.get("model_path", "transformer_model.h5")
TOKENIZER_PATH = CONFIG.get("tokenizer_path", "tokenizer.json")
LABEL_ENCODER_PATH = CONFIG.get("label_encoder_path", "label_encoder.json")

def load_model_and_tokenizer():
    model = load_model(MODEL_PATH, custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "TransformerBlock": TransformerBlock
    })

    with open(TOKENIZER_PATH, "r") as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)

    with open(LABEL_ENCODER_PATH, "r") as f:
        label_encoder = json.load(f)

    return model, tokenizer, label_encoder

def predict_sentiment(text, model, tokenizer, label_encoder):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0]
    label = label_encoder[np.argmax(pred)]
    return label
