import torch
import torch.nn as nn

# Replace this with your actual model class from the notebook
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Define architecture same as your training
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.network(x)

def load_model(model_path="diffusion_model.pth"):
    model = DiffusionModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model