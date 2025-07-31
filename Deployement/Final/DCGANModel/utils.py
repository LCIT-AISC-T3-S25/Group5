import torch
import torchvision.utils as vutils
import os
from datetime import datetime

def generate_image(generator, device, nz=100):
    noise = torch.randn(1, nz, 1, 1, device=device)
    with torch.no_grad():
        fake_image = generator(noise).detach().cpu()
    return fake_image

def save_image(tensor, save_dir="static"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path = f"{save_dir}/generated_{timestamp}.png"
    vutils.save_image(tensor, path, normalize=True)
    return path
