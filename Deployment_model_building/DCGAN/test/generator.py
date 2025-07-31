import torch.nn as nn
import json

# Load config
with open("config.json") as f:
    config = json.load(f)

gen_cfg = config.get("generator_config", {})

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        nz = gen_cfg.get("nz", 100)
        fmap_list = gen_cfg.get("feature_maps", [256, 128, 64])
        output_channels = gen_cfg.get("output_channels", 3)
        kernel = gen_cfg.get("kernel_size", 4)
        stride = gen_cfg.get("stride", 2)
        padding = gen_cfg.get("padding", 1)

        layers = []

        # First layer: latent vector to first feature map
        layers.append(nn.ConvTranspose2d(nz, fmap_list[0], kernel, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(fmap_list[0]))
        layers.append(nn.ReLU(True))

        # Hidden layers
        for i in range(len(fmap_list) - 1):
            layers.append(nn.ConvTranspose2d(fmap_list[i], fmap_list[i + 1], kernel, stride, padding, bias=False))
            layers.append(nn.BatchNorm2d(fmap_list[i + 1]))
            layers.append(nn.ReLU(True))

        # Output layer
        layers.append(nn.ConvTranspose2d(fmap_list[-1], output_channels, kernel, stride, padding, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)
