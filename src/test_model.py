import torch
import torch.nn as nn
from torchvision import models
import pickle
import numpy as np
import os


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
        return nn.functional.softmax(x, dim=1)


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
weights_path = os.path.join(base_dir, "dropout", "aesthetic_rater.pt")
test_label_path = os.path.join(
    base_dir, "dropout", "test_label_distribution.dat")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BeautyResNet().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

score, pred_score = [], []
test_label_dist = pickle.load(open(test_label_path, 'rb'))
for label_dist in test_label_dist:
    image = label_dist[1].astype(np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    label_score = sum((i+1)*v for i, v in enumerate(label_dist[2]))
    print("Test image name:" + str(label_dist[0]))
    print("Annotated score:%1.2f " % label_score)
    score.append(label_score)
    with torch.no_grad():
        pred = model(image)[0].cpu().numpy()
    pred_val = sum((i+1)*v for i, v in enumerate(pred))
    print("Predicted score:" + str(pred_val))
    pred_score.append(pred_val)

y = np.asarray(score)
pred_y = np.asarray(pred_score)
if len(set(pred_score)) <= 1 or len(set(score)) <= 1:
    print("Predicted or true scores are constant. Pearson correlation is undefined.")
else:
    corr = np.corrcoef(y, pred_y)[0, 1]
    print('PC (Pearson correlation) mean = %1.2f ' % corr)
