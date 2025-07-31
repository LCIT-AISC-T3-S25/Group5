import torch
import json

# Load config
with open("config.json") as f:
    config = json.load(f)

gen_cfg = config.get("generation_config", {})
nz = gen_cfg.get("nz", 100)

# Generate image tensor using random noise
def generate_image(generator, device):
    noise = torch.randn(1, nz, 1, 1, device=device)
    with torch.no_grad():
        fake_image = generator(noise).detach().cpu()
    return fake_image
