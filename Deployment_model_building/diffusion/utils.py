import torch
import torch.nn as nn
import clip

class DiffusionModel(nn.Module):
    def __init__(self, clip_model, embedding_dim=128):
        super().__init__()
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False

        self.time_embed = nn.Embedding(1000, embedding_dim)

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

        self.text_proj = nn.Linear(512, 256)
        self.time_proj = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.text_to_128 = nn.Linear(256, 128)
        self.time_to_128 = nn.Linear(256, 128)

    def forward(self, x, t, captions):
        with torch.no_grad():
            text_features = self.clip.encode_text(captions).float()

        text_embed = self.text_proj(text_features)
        time_embed = self.time_proj(self.time_embed(t))

        text_embed_128 = self.text_to_128(text_embed).unsqueeze(-1).unsqueeze(-1)
        time_embed_128 = self.time_to_128(time_embed).unsqueeze(-1).unsqueeze(-1)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x = x2 + text_embed_128 + time_embed_128
        x = self.up1(x)
        x = self.up2(x)
        return x

def denormalize(image_tensor):
    return (image_tensor * 0.5 + 0.5).clamp(0, 1)
