import json
from flask_wtf.csrf import CSRFProtect
from app import app  # make sure app.py is the correct module name

def test_health_check():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"healthy" in response.data

def test_predict_valid(monkeypatch):
    client = app.test_client()

    # Fake tokenizer and label encoder
    class DummyTokenizer:
        def texts_to_sequences(self, texts):
            return [[1, 2, 3]]

    class DummyModel:
        def __call__(self, x):
            import numpy as np
            return np.array([[0.1, 0.9]])

    class DummyEncoder:
        def inverse_transform(self, y):
            return ["positive"]

    # Monkeypatching
    app.tokenizer = DummyTokenizer()
    app.model = DummyModel()
    app.label_encoder = DummyEncoder()

    payload = {"text": "Sample text"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert b"predicted_sentiment" in response.data

def test_predict_missing_text():
    client = app.test_client()
    response = client.post("/predict", json={})
    assert response.status_code == 400
