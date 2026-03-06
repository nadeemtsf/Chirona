# SIGNSENSE — Full Repo Context Dump
# Real-Time Sign Language Interpreter (ASL → Text/Speech)
# By Michael Musallam and Nadim Baboun
# Created: Feb 27, 2026

## PROJECT OVERVIEW

Python-based hand tracking + ASL sign language recognition system using
MediaPipe, OpenCV, and scikit-learn. Two modes: gesture mouse control and
sign language interpretation. Currently has 2 classes (a, b) trained with
a RandomForest classifier achieving 100% accuracy on test set.

Tech: Python 3.9-3.11, OpenCV, MediaPipe, scikit-learn, pyautogui, seaborn, matplotlib

## PROJECT STRUCTURE

```
hand-2-cursor/
├── main.py                          # Entry point — webcam loop, mode switching (m key)
├── config.py                        # All constants (camera, gestures, classes, augmentation)
├── requirements.txt
├── README.md
├── ROADMAP.md                       # Detailed dev roadmap (phases 0–7)
├── core/
│   ├── hand_detector.py             # MediaPipe hand detection wrapper (21 landmarks)
│   ├── feature_extractor.py         # STUB — "landmark -> feature vector"
│   ├── sign_classifier.py           # STUB — "sign language classifier file"
│   └── sentence_builder.py          # STUB — "word/sentence accumulation"
├── controllers/
│   ├── mouse_controller.py          # Gesture mouse: move, L/R click, scroll via pinch
│   └── sign_language_controller.py  # Placeholder — draws hand, shows "classifier not loaded"
├── utils/
│   ├── drawing_utils.py             # Hand points, skeleton, bounding box drawing
│   ├── text_overlay.py              # STUB — "text display on frame"
│   └── augment.py                   # Image augmentation (flip, rotate, brightness, zoom)
├── data/
│   ├── collect_images.py            # Webcam image capture per letter class
│   ├── extract_landmarks.py         # Images → normalized 42-feature vectors → landmarks.pickle
│   ├── augment_dataset.py           # Runs augmentation pipeline on raw images
│   ├── load_asl_mnist.py            # ASL MNIST CSV loader with self-test
│   ├── verify_data.py               # Data quality checker (shapes, NaN, imbalance)
│   ├── raw/                         # Raw captured images (a-z folders, minus j/z)
│   │   └── a/, b/, c/, ... z/
│   └── asl_mnist/                   # Kaggle ASL MNIST CSVs
│       ├── sign_mnist_train.csv
│       └── sign_mnist_test.csv
├── models/
│   ├── train_model.py               # Train RandomForest, save to models/saved/model_rf.p
│   ├── evaluate_model.py            # Evaluate model: accuracy, report, confusion matrix, top confused pairs
│   └── saved/
│       └── model_rf.p               # Trained RF model (currently 2 classes: a, b)
```

## CONFIG (config.py)

```python
CAMERA_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
SMOOTHING_ALPHA = 0.3
FRAME_REDUCTION = 150
PINCH_DISTANCE = 40
CLICK_COOLDOWN = 0.1
SCROLL_JITTER_THRESHOLD = 5
SCROLL_SPEED_MULTIPLIER = 1.5
FINGER_CIRCLE_RADIUS = 15
WINDOW_TITLE = "Hand Tracking"
CLASSES = list("abcdefghiklmnopqrstuvwxy")  # 24 letters (no J, no Z)
IMAGES_PER_CLASS = 50
COUNTDOWN = 3
DELAY_BETWEEN_CLASSES = 50
NUM_CLASSES = len(CLASSES)
DATA_DIR = "./data/raw"
ASL_MNIST_DIR = "data/asl_mnist"
AUGMENT_CONFIG = {"flip": True, "rotate": True, "brightness": True, "zoom": False, "factor": 5}
```

## COMPLETE SOURCE CODE

### main.py

```python
import cv2
import time
from core.hand_detector import HandDetector
from controllers.mouse_controller import MouseController
from controllers.sign_language_controller import SignLanguageController
from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    SMOOTHING_ALPHA, WINDOW_TITLE,
)

controllers = {
    "mouse": MouseController(alpha=SMOOTHING_ALPHA),
    "sign_language": SignLanguageController(),
}
detector = HandDetector(MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

prev_time = 0
mode = "mouse"

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    hands_data = detector.detect(frame)

    if hands_data:
        controllers[mode].process_frame(frame, hands_data[0], detector)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Mode: {mode}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(WINDOW_TITLE, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "sign_language" if mode == "mouse" else "mouse"
    if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
```

### core/hand_detector.py

```python
import cv2
import mediapipe as mp
from config import MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE

class HandDetector:
    def __init__(self, max_hands=MAX_HANDS, detection_confidence=DETECTION_CONFIDENCE, tracking_confidence=TRACKING_CONFIDENCE):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def detect(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_info.classification[0].label
                positions = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    positions.append((idx, px, py))
                hands_data.append({
                    "label": hand_label,
                    "landmarks": hand_landmarks,
                    "positions": positions
                })
        return hands_data
```

### core/feature_extractor.py

```python
# landmark -> feature vector (STUB)
```

### core/sign_classifier.py

```python
# sign language classifier file (STUB)
```

### core/sentence_builder.py

```python
# word/sentence accumulation (STUB)
```

### controllers/mouse_controller.py

```python
import time, math, cv2
import numpy as np
import pyautogui
from utils.drawing_utils import draw_hand_points, draw_hand_skeleton
from config import (CAM_WIDTH, CAM_HEIGHT, FRAME_REDUCTION, PINCH_DISTANCE,
                    SCROLL_JITTER_THRESHOLD, SCROLL_SPEED_MULTIPLIER,
                    CLICK_COOLDOWN, FINGER_CIRCLE_RADIUS)

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

class MouseController:
    def __init__(self, alpha=0.25, move_interval=0.01, dead_zone=3):
        self.screen_w, self.screen_h = pyautogui.size()
        self.current_x, self.current_y = pyautogui.position()
        self.alpha = alpha
        self.dead_zone = dead_zone
        self.move_interval = move_interval
        self.last_move_time = 0
        self.prev_y1 = 0

    def process_frame(self, frame, hand, detector):
        positions = hand["positions"]
        hand_landmarks = hand["landmarks"]
        draw_hand_points(frame, positions)
        draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)
        if not positions: return
        x1, y1 = positions[8][1], positions[8][2]   # Index tip
        x2, y2 = positions[4][1], positions[4][2]   # Thumb tip
        x3, y3 = positions[12][1], positions[12][2] # Middle tip
        x4, y4 = positions[16][1], positions[16][2] # Ring tip
        dist_scroll = math.hypot(x2 - x4, y2 - y4)
        if dist_scroll < PINCH_DISTANCE:
            cv2.circle(frame, (x4, y4), FINGER_CIRCLE_RADIUS, (255, 255, 0), cv2.FILLED)
            if self.prev_y1 == 0: self.prev_y1 = y1
            delta_y = y1 - self.prev_y1
            if abs(delta_y) > SCROLL_JITTER_THRESHOLD:
                self.scroll(int(-delta_y * SCROLL_SPEED_MULTIPLIER))
        else:
            cv2.circle(frame, (x1, y1), FINGER_CIRCLE_RADIUS, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), FINGER_CIRCLE_RADIUS, (255, 0, 255), cv2.FILLED)
            x_screen = np.interp(x1, (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, self.screen_w))
            y_screen = np.interp(y1, (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, self.screen_h))
            self.move(x_screen, y_screen)
            if math.hypot(x2 - x1, y2 - y1) < PINCH_DISTANCE:
                cv2.circle(frame, (x1, y1), FINGER_CIRCLE_RADIUS, (0, 255, 0), cv2.FILLED)
                self.click('left'); time.sleep(CLICK_COOLDOWN)
            if math.hypot(x2 - x3, y2 - y3) < PINCH_DISTANCE:
                cv2.circle(frame, (x3, y3), FINGER_CIRCLE_RADIUS, (0, 0, 255), cv2.FILLED)
                self.click('right'); time.sleep(CLICK_COOLDOWN)
        self.prev_y1 = y1

    def move(self, x, y):
        now = time.time()
        if now - self.last_move_time < self.move_interval: return
        self.last_move_time = now
        x = max(0, min(self.screen_w - 1, x))
        y = max(0, min(self.screen_h - 1, y))
        if abs(x - self.current_x) < self.dead_zone and abs(y - self.current_y) < self.dead_zone: return
        self.current_x = self.current_x * (1 - self.alpha) + x * self.alpha
        self.current_y = self.current_y * (1 - self.alpha) + y * self.alpha
        pyautogui.moveTo(int(self.current_x), int(self.current_y))

    def click(self, button="left"): pyautogui.click(button=button)
    def scroll(self, dy): pyautogui.scroll(dy)
```

### controllers/sign_language_controller.py

```python
import cv2
from utils.drawing_utils import draw_hand_points, draw_hand_skeleton

class SignLanguageController:
    def __init__(self): pass

    def process_frame(self, frame, hand, detector):
        positions = hand["positions"]
        hand_landmarks = hand["landmarks"]
        draw_hand_points(frame, positions)
        draw_hand_skeleton(frame, hand_landmarks, detector.mp_hands, detector.mp_drawing)
        cv2.putText(frame, "Sign Language Mode - classifier not loaded yet",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
```

### utils/drawing_utils.py

```python
import cv2
TIP_IDS = [4, 8, 12, 16, 20]

def draw_hand_points(frame, positions, tip_ids=TIP_IDS):
    for idx, px, py in positions:
        radius = 8 if idx in tip_ids else 5
        cv2.circle(frame, (px, py), radius, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (px, py), radius, (0, 0, 0), 1)

def draw_bounding_box(frame, positions, label, color):
    x_coords = [p[1] for p in positions]
    y_coords = [p[2] for p in positions]
    x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
    y_min, y_max = min(y_coords) - 20, max(y_coords) + 20
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.rectangle(frame, (x_min, y_min - 30), (x_min + 120, y_min), color, cv2.FILLED)
    cv2.putText(frame, label, (x_min + 5, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_hand_skeleton(frame, hand_landmarks, mp_hands, mp_drawing):
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
```

### utils/text_overlay.py

```python
# text display on frame (STUB)
```

### utils/augment.py

```python
import cv2, numpy as np, random
from pathlib import Path

def horizontal_flip(img): return cv2.flip(img, 1)

def random_rotation(img, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def brightness_contrast(img):
    alpha = 1 + random.uniform(-0.2, 0.2)
    beta = random.uniform(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def random_zoom(img):
    h, w = img.shape[:2]
    zoom = 1 + random.uniform(-0.1, 0.1)
    new_h, new_w = int(h / zoom), int(w / zoom)
    y1, x1 = (h-new_h)//2, (w-new_w)//2
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h))

def apply_augmentations(img, config):
    out = img.copy()
    if config["flip"]: out = horizontal_flip(out)
    if config["rotate"]: out = random_rotation(out)
    if config["zoom"]: out = random_zoom(out)
    if config["brightness"]: out = brightness_contrast(out)
    return out

def augment_dataset(input_root, output_root, factor, config):
    input_root, output_root = Path(input_root), Path(output_root)
    total_original, total_augmented = 0, 0
    for class_dir in sorted(input_root.iterdir()):
        if not class_dir.is_dir(): continue
        out_class_dir = output_root / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        images = list(class_dir.glob("*.jpg"))
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None: continue
            total_original += 1
            cv2.imwrite(str(out_class_dir / img_path.name), img)
            for i in range(factor - 1):
                aug = apply_augmentations(img, config)
                cv2.imwrite(str(out_class_dir / f"{img_path.stem}_aug{i}.png"), aug)
                total_augmented += 1
    print(f"original: {total_original}, augmented: {total_augmented}, total: {total_original + total_augmented}")
```

### data/collect_images.py

```python
# Webcam capture script. Opens camera, for each letter in CLASSES:
# 1. Waits for user to press 'S' to start
# 2. 3-second countdown
# 3. Captures IMAGES_PER_CLASS (50) frames to data/raw/{letter}/
# Press 'Q' to quit at any time. Images saved as {timestamp}.jpg

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import time
from pathlib import Path
from config import (CLASSES, IMAGES_PER_CLASS, CAM_HEIGHT, CAM_WIDTH,
                    COUNTDOWN, DELAY_BETWEEN_CLASSES, DATA_DIR, CAMERA_INDEX)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

for class_letter in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_letter)
    os.makedirs(class_dir, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret: exit(1)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Letter {class_letter.upper()} - Press "S" to start', (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Collect', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        if key == ord('s'): break
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: exit(1)
        frame = cv2.flip(frame, 1)
        remaining = COUNTDOWN - int(time.time() - start_time)
        if remaining <= 0: break
        cv2.putText(frame, f'{remaining + 1}', (CAM_WIDTH//2 - 50, CAM_HEIGHT//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 4)
        cv2.imshow('Collect', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): cap.release(); cv2.destroyAllWindows(); sys.exit(0)
    for img_num in range(IMAGES_PER_CLASS):
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Letter {class_letter.upper()} - Image {img_num+1}/{IMAGES_PER_CLASS}',
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Collect', frame)
        cv2.imwrite(os.path.join(class_dir, f'{int(time.time() * 1000)}.jpg'), frame)
        if cv2.waitKey(DELAY_BETWEEN_CLASSES) & 0xFF == ord('q'):
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)

cap.release()
cv2.destroyAllWindows()
```

### data/extract_landmarks.py

```python
# Processes all images in data/raw/. For each image:
# 1. Detects hand with MediaPipe (static_image_mode=True)
# 2. Computes bounding box around 21 landmarks
# 3. Normalizes landmarks relative to bbox → 42 features (21 x,y)
# 4. Saves to data/landmarks.pickle as {"data": np.float32 array, "labels": np.array of strings}

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from core.hand_detector import HandDetector
from config import MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE

def get_hand_box(landmarks):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    return min(xs), min(ys), max(xs), max(ys)

def normalize_landmarks(landmarks, box):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min if x_max - x_min != 0 else 1e-6
    height = y_max - y_min if y_max - y_min != 0 else 1e-6
    normalized = []
    for lm in landmarks.landmark:
        normalized.append((lm.x - x_min) / width)
        normalized.append((lm.y - y_min) / height)
    return normalized

def extract_landmarks_batch(raw_data_dir="./data/raw", output_path="./data/landmarks.pickle"):
    detector = HandDetector(MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE)
    all_data, all_labels = [], []
    total_images, successful, skipped = 0, 0, 0
    root = Path(raw_data_dir)
    if not root.exists(): return
    class_folders = [d for d in root.iterdir() if d.is_dir()]
    detector.hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=MAX_HANDS,
        min_detection_confidence=DETECTION_CONFIDENCE, min_tracking_confidence=TRACKING_CONFIDENCE)
    for class_folder in class_folders:
        class_label = class_folder.name
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        for image_path in images:
            total_images += 1
            image = cv2.imread(str(image_path))
            if image is None: skipped += 1; continue
            hands_data = detector.detect(image)
            if not hands_data: skipped += 1; continue
            for hand in hands_data:
                landmarks = hand['landmarks']
                box = get_hand_box(landmarks)
                normalized = normalize_landmarks(landmarks, box)
                all_data.append(normalized)
                all_labels.append(class_label)
                successful += 1
    if successful == 0: return
    all_data = np.array(all_data, dtype=np.float32)
    all_labels = np.array(all_labels)
    with open(output_path, "wb") as f:
        pickle.dump({"data": all_data, "labels": all_labels}, f)
    logging.info(f"Total: {total_images}, Success: {successful}, Skipped: {skipped}, Shape: {all_data.shape}")

if __name__ == "__main__":
    extract_landmarks_batch()
```

### data/augment_dataset.py

```python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.augment import augment_dataset
from config import DATA_DIR, AUGMENT_CONFIG

if __name__ == "__main__":
    augment_dataset(input_root=DATA_DIR, output_root=DATA_DIR,
                    factor=AUGMENT_CONFIG["factor"], config=AUGMENT_CONFIG)
```

### data/load_asl_mnist.py

```python
# Loads Kaggle ASL MNIST CSVs (28x28 grayscale images, labels 0-25, no J/Z)
# API: load_asl_mnist(split="train"|"test"|"both", flatten=bool, normalize=bool)
# Returns (images, labels, label_map) or (train_imgs, train_lbls, test_imgs, test_lbls, label_map)
# Also has save_as_images() to convert CSV to per-letter PNG folders
# Self-test: python -m data.load_asl_mnist

import sys, os, csv
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ASL_MNIST_DIR

LABEL_TO_LETTER = {i: chr(ord("A") + i) for i in range(26)}
VALID_LABELS = sorted(set(range(26)) - {9, 25})  # no J(9) or Z(25)

def _csv_to_arrays(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([int(v) for v in row])
    data = np.array(rows, dtype=np.int64)
    return data[:, 1:].astype(np.uint8).reshape(-1, 28, 28), data[:, 0]

def load_asl_mnist(data_dir=None, split="train", flatten=False, normalize=False):
    data_dir = Path(data_dir) if data_dir else Path(ASL_MNIST_DIR)
    file_map = {"train": "sign_mnist_train.csv", "test": "sign_mnist_test.csv"}
    label_map = {i: LABEL_TO_LETTER[i] for i in VALID_LABELS}
    def _load(name):
        imgs, lbls = _csv_to_arrays(data_dir / file_map[name])
        if flatten: imgs = imgs.reshape(imgs.shape[0], -1)
        if normalize: imgs = imgs.astype(np.float32) / 255.0
        return imgs, lbls
    if split == "both":
        tr_imgs, tr_lbls = _load("train")
        te_imgs, te_lbls = _load("test")
        return tr_imgs, tr_lbls, te_imgs, te_lbls, label_map
    imgs, lbls = _load(split)
    return imgs, lbls, label_map
```

### data/verify_data.py

```python
# Validates data/landmarks.pickle for common issues
# Checks: structure, shapes, NaN/Inf, class imbalance, value ranges
# Usage: python -m data.verify_data [path] [--fix]
# --fix removes bad samples and resaves. Exit code 0=pass, 1=fail
# Expected: data shape (N, 42), labels are strings, values in ~[0,1]
```

### models/train_model.py

```python
import pickle, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "landmarks.pickle")

if not os.path.exists(data_path):
    print("Error: Run python data/extract_landmarks.py first.")
    exit(1)

with open(data_path, "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset["data"])
y = np.array(dataset["labels"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {accuracy:.2f}")

model_dir = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, "model_rf.p"), "wb") as f:
    pickle.dump(model, f)
```

### models/evaluate_model.py

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, class_labels=None, save_path=None):
    y_pred = model.predict(X_test)
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    if class_labels is not None:
        label_set = set(str(l) for l in unique_labels)
        display_labels = [l for l in class_labels if l in label_set]
        if not display_labels:
            display_labels = [str(l) for l in unique_labels]
    else:
        display_labels = [str(l) for l in unique_labels]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}\n")

    report = classification_report(y_test, y_pred, labels=unique_labels, target_names=display_labels)
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    print("Classification Report:")
    print(report)

    labels = display_labels
    confused_pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((labels[i], labels[j], cm[i][j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_confused = confused_pairs[:5]

    print("Top 5 Most-Confused Class Pairs:")
    for true_label, pred_label, count in top_confused:
        print(f"  True: {true_label} -> Predicted: {pred_label} ({count} times)")

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_labels, yticklabels=display_labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    if save_path: plt.savefig(save_path)
    plt.show()

    return {'accuracy': accuracy, 'classification_report': report,
            'confusion_matrix': cm, 'top_confused_pairs': top_confused}
```

## CURRENT STATE

- Data pipeline: WORKING (collect → extract → landmarks.pickle)
- Only 2 classes collected so far (a, b) with 50 images each
- Model: RandomForest, 100% accuracy on a/b test set
- Mouse control mode: FULLY WORKING
- Sign language mode: PLACEHOLDER (draws hand, no classification yet)
- STUBS remaining: feature_extractor.py, sign_classifier.py, sentence_builder.py, text_overlay.py
- ASL MNIST data available but not yet integrated into training pipeline

## KEY DATA FORMATS

- landmarks.pickle: {"data": np.float32 (N, 42), "labels": np.array of strings}
- 42 features = 21 landmarks x 2 (x, y), normalized to hand bounding box [0, 1]
- Labels are lowercase letter strings ("a", "b", ...)
- model_rf.p: pickled sklearn RandomForestClassifier

## DEPENDENCIES (requirements.txt)

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pyautogui
scikit-learn>=1.3.0
tensorflow>=2.13.0
pyttsx3
```