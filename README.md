# AestheticRater

Predicts facial beauty scores using PyTorch and ResNet50.

## Step-by-step Usage

**1. Clone the repository**

```
git clone https://github.com/yourusername/AestheticRater.git
cd AestheticRater
```

**2. Install dependencies**

```
pip install -r requirements.txt
```

**3. Prepare data**

- Download SCUT-FBP5500 images from [Google Drive](https://drive.usercontent.google.com/download?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf&authuser=0) and extract to `ref/SCUT-FBP5500_v2/`
- Download YOLOv8 face model from [lindevs/yolov8-face](https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt) and place in `ref/yolov8n-face-lindevs.pt`

**4. Prepare label distributions**

```
python src/prepare_data.py
```

**5. Train the model**

```
python src/train_model.py
```

**6. Evaluate the model**

```
python src/test_model.py
```

**7. Run inference on images**

```
python inference/aesthetic_rater.py
```

## Files

- `src/prepare_data.py` – Prepares face crops and label distributions.
- `src/train_model.py` – Trains the model.
- `src/test_model.py` – Evaluates on test set.
- `inference/aesthetic_rater.py` – Predicts and annotates images.

## Requirements

- Python 3.13+
- torch, torchvision, torchaudio
- opencv-python
- pandas, numpy
- tqdm
- pillow
- ultralytics
- bing-image-downloader
- openpyxl
- scipy

## Notes

- All paths are relative to the project root.
- Data files must be present before running.
- For a full list of packages: `pip list`
