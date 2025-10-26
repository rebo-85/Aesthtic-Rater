import cv2
import os
import pickle
import numpy as np
import random
from PIL import Image, ImageEnhance
import pandas as pd
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "ref", "SCUT-FBP5500_v2", "Images")
out_path = os.path.join(base_dir, "dropout")
xlsx_path = os.path.join(
    base_dir, "ref", "SCUT-FBP5500_v2", "All_Ratings.xlsx")

yolo_face = YOLO("ref/yolov8n-face-lindevs.pt")


def detect_face_yolo(model, image_path, image_name):
    imgAbsPath = os.path.join(image_path, image_name)
    img = cv2.imread(imgAbsPath)
    if img is None:
        print(image_name + " not found or unreadable")
        return None
    results = model(img)
    faces = results[0].boxes.xyxy.cpu().numpy(
    ) if results and results[0].boxes is not None else []
    if len(faces) == 1:
        x1, y1, x2, y2 = faces[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        h, w = img.shape[:2]
        pad = int(0.3 * max(x2 - x1, y2 - y1))
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad)
        y2p = min(h, y2 + pad)
        crop = img[y1p:y2p, x1p:x2p]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            print("invalid shape")
            return None
        resized = cv2.resize(crop, (224, 224))
        return resized
    print(image_name + " error " + str(len(faces)))
    return None


def randomUpdate(img):
    img = Image.fromarray(img.astype(np.uint8))
    rotate = random.random() * 30 - 30
    image_rotated = img.rotate(rotate)
    enh_bri = ImageEnhance.Brightness(image_rotated)
    bright = random.random() * 0.8 + 0.6
    image_brightened = enh_bri.enhance(bright)
    enh_con = ImageEnhance.Contrast(image_brightened)
    contrast = random.random() * 0.6 + 0.7
    image_contrasted = enh_con.enhance(contrast)
    enh_col = ImageEnhance.Color(image_contrasted)
    color = random.random() * 0.6 + 0.7
    image_colored = enh_col.enhance(color)
    return np.asarray(image_colored)


label_distribution = []
df = pd.read_excel(xlsx_path)
grouped = df.groupby('Filename')
for image_name, group in grouped:
    score_counts = group['Rating'].value_counts().sort_index()
    total_votes = score_counts.sum()
    ld = [(score_counts.get(i, 0) / total_votes) for i in range(1, 6)]
    im = detect_face_yolo(yolo_face, data_path, image_name)
    if isinstance(im, np.ndarray):
        normed_im = (im - 127.5) / 127.5
        label_distribution.append([image_name, normed_im, ld])
    else:
        print(image_name + " face not detected, sample dropped")

split_idx = int(len(label_distribution) - len(label_distribution) * 0.1)
random.shuffle(label_distribution)
test_label_distribution = label_distribution[split_idx:]
train_label_distribution = label_distribution[:split_idx]

for i in range(len(train_label_distribution)):
    img_name, img_arr, label_dist = train_label_distribution[i]
    aug_img = randomUpdate(img_arr)
    aug_img_norm = (aug_img - 127.5) / 127.5
    train_label_distribution.append(
        [img_name + "_aug", aug_img_norm, label_dist])

if not os.path.exists(out_path):
    os.makedirs(out_path)

print("Generating train_label_distribution.dat")
random.shuffle(train_label_distribution)
pickle.dump(train_label_distribution, open(
    os.path.join(out_path, "train_label_distribution.dat"), 'wb'))
print("Generating test_label_distribution.dat")
random.shuffle(test_label_distribution)
pickle.dump(test_label_distribution, open(
    os.path.join(out_path, "test_label_distribution.dat"), 'wb'))
