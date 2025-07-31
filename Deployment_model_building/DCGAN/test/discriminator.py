import torch.nn as nn
import json

# Load config.json once
with open("config.json") as f:
    config = json.load(f)

disc_cfg = config.get("discriminator_config", {})

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = disc_cfg.get("input_channels", 3)
        fmap_list = disc_cfg.get("feature_maps", [128, 256, 512, 1024])
        kernel = disc_cfg.get("kernel_size", 4)
        stride = disc_cfg.get("stride", 2)
        padding = disc_cfg.get("padding", 1)
        slope = disc_cfg.get("leaky_relu_slope", 0.2)

        layers = []
        for i, out_channels in enumerate(fmap_list):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False))
            if i > 0:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(slope, inplace=True))
            in_channels = out_channels

        # Final layer
        layers.append(nn.Conv2d(fmap_list[-1], 1, kernel, 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, img):
        return self.main(img).view(-1)
