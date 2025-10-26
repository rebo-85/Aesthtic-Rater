import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import os
import cv2
from ultralytics import YOLO
import re
from bing_image_downloader import downloader
import shutil

queries = [
    "headshot sample",
    "headshot asian",
    "headshot african",
    "headshot indian",
    "headshot caucasian"
]

samples_dir = 'inference/samples'
download_prompt = input("Download new images? (y/n): ").strip().lower()
if download_prompt == "y":
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)
    os.makedirs(samples_dir, exist_ok=True)
    overall_limit = 50
    per_query = max(1, overall_limit // len(queries))
    for q in queries:
        safe_query = re.sub(r'[\\/:*?"<>|]', '_', q.replace(" ", "_"))
        downloader.download(safe_query, limit=per_query, output_dir='inference/samples',
                            adult_filter_off=True, force_replace=False, timeout=60)


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
weights_path = os.path.join(
    base_dir, "dropout", "aesthetic_rater.inference.pt")
samples_dir = os.path.join(base_dir, "inference", "samples")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BeautyResNet().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

face_detector = YOLO(os.path.join(base_dir, "ref", "yolov8n-face-lindevs.pt"))
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
SCORED_SUFFIXES = tuple(f'_scored{ext}' for ext in IMAGE_EXTS)

for root, _, files in os.walk(samples_dir):
    for fname in files:
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        if any(fname.endswith(s) for s in SCORED_SUFFIXES):
            continue
        img_path = os.path.join(root, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = face_detector(img)
        face_box = None
        if faces and hasattr(faces[0], "boxes") and faces[0].boxes is not None and len(faces[0].boxes) > 0:
            box = faces[0].boxes.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, box)
            h, w = img.shape[:2]
            pad = int(0.30 * max(x2 - x1, y2 - y1))
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)
            face_box = (x1p, y1p, x2p, y2p)
            crop = img[y1p:y2p, x1p:x2p]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                crop = img
        else:
            crop = img

        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        normed = (img_resized.astype(np.float32) - 127.5) / 127.5
        img_tensor = torch.from_numpy(normed).permute(
            2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)[0].cpu().numpy()
        score = sum((i+1)*v for i, v in enumerate(pred))

        img_draw = img.copy()
        if face_box:
            x1p, y1p, x2p, y2p = face_box
            cv2.rectangle(img_draw, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            txt_x = x1p
            txt_y = y2p - 5
            cv2.putText(img_draw, f"{score:.2f}", (txt_x, txt_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(img_draw, f"{score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        name, ext = os.path.splitext(fname)
        scored_path = os.path.join(root, f"{name}_scored{ext}")
        cv2.imwrite(scored_path, img_draw)
        print(f"Score: {score:.2f}, Saved at: {scored_path}")
