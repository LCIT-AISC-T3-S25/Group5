from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
from model.cnn_model import SimpleCNN  # Adjust if your model is different

# ✅ This is the key line — FastAPI instance
app = FastAPI()

# Load model
model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load("cnn_model_rgb.h5", map_location="cpu"))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return {"predicted_class": int(predicted.item())}
