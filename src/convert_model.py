import torch
from torchvision import models
import torch.nn as nn
import os
import onnx


class BeautyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, 5)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # logits, apply softmax in JS if needed


# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights_path = os.path.join(base_dir, "dropout", "aesthetic_rater.pt")
onnx_path = os.path.join(base_dir, "dropout", "aesthetic_rater.onnx")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BeautyResNet().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# Export PyTorch -> ONNX
dummy_input = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
    opset_version=18
)

print("Conversion complete! ONNX model is in:", onnx_path)
