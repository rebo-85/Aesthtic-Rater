import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.transforms import Normalize
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
from tqdm import tqdm
import platform

imagenet_norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class AestheticDataset(Dataset):
    def __init__(self, label_distribution):
        self.images = [x[1] for x in label_distribution]
        self.labels = [x[2] for x in label_distribution]
        self.images = np.stack(self.images).astype(np.float32)
        self.labels = np.stack(self.labels).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        return img, torch.from_numpy(self.labels[idx])


class AestheticResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, 5)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if "layer2" in name or "layer3" in name or "layer4" in name:
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)


train_label_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "dropout", "train_label_distribution.dat")
if not os.path.exists(train_label_path):
    print('train_label_distribution.dat not found')
    exit(1)

with open(train_label_path, 'rb') as f:
    try:
        label_dist = pickle.load(f)
        print('len:', len(label_dist))
        for i, item in enumerate(label_dist[:5]):
            print(f'{i}:', item)
    except Exception as e:
        print('Error loading:', e)
        exit(1)

if not label_dist:
    print('train_label_distribution.dat is empty or corrupted')
    exit(1)

dataset = AestheticDataset(label_dist)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AestheticResNet().to(device)
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       model.parameters()), lr=0.0005)

model_path = 'dropout/aesthetic_rater.pt'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))


def get_system_info():
    cpu = platform.processor()
    if not cpu:
        cpu = platform.uname().processor or platform.uname().machine
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return cpu, gpu


cpu_model, gpu_model = get_system_info()

print('CPU:', cpu_model)
print('GPU:', gpu_model if gpu_model else 'None')

if gpu_model and device.type == 'cpu':
    print('Warning: CUDA-capable GPU detected but script is running on CPU.')

best_loss = float('inf')
patience = 10
epochs = 100
wait = 0

print(f"Training started")

try:
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'dropout/aesthetic_rater.pt')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping")
                break

        print(
            f"| Loss: {epoch_loss:.4f} | Best Loss: {best_loss:.4f} |")
        torch.save(model.state_dict(), 'dropout/aesthetic_rater.inference.pt')
except BaseException as e:
    print('Training interrupted:', e)
    raise
