import numpy as np
from functools import lru_cache
from PIL import Image
from ultralytics import YOLO

from image_utils import annotate_image

BINARY_THRESHOLD = 0.5 # minimum L1 confidence to be called "pressure sore"
SEVERITY_THRESHOLD = 0.0 # set > 0 if you want a minimum confidence at L2/L3

# Level 1 — PS vs No-PS 
L1_MODEL_PATHS = [
    "models/yolo_cascade/best_YoloV26(yolo26n-cls.pt) PS or Not PS Feb 24  2026 0.9962.pt",
    "models/yolo_cascade/best_YoloV8(yolov8n-cls.pt) PS or Not PS Feb 24  2026 1.0.pt",
]

# Level 2 — Early (Stage I or II) vs Advanced (Stage III or IV)
L2_MODEL_PATHS = [
    "models/yolo_cascade/best_YoloV8(yolov8s-cls.pt) E vs A PS Feb 23  2026 0.9718.pt",
    "models/yolo_cascade/best_Yolo26(yolo26s-cls.pt) E vs A PS Feb 23  2026 0.9637.pt",
]

# Level 3a — Early group: Stage I vs Stage II
L3_EARLY_MODEL_PATHS = [
    "models/yolo_cascade/best_YoloV8(yolov8n-cls.pt) Early PS Feb 23  2026 0.9032.pt",
    "models/yolo_cascade/best_YoloV8(yolov8x-cls.pt) Early PS Feb 23  2026 0.9032.pt",
]

# Level 3b — Advanced group: Stage III vs Stage IV
L3_ADVANCED_MODEL_PATHS = [
    "models/yolo_cascade/best_YoloV11(yolo11m-cls.pt) Advanced PS Feb 22  2026 0.7742.pt",
    "models/yolo_cascade/best_YoloV8(yolov8n-cls.pt) Advanced PS Feb 22  2026 0.8226.pt",
]

# Label normalization helper
def _norm(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").strip()

# Lazy model loading: one cached loader per cascade level so each level's
# models are loaded only once regardless of how many requests come in.

@lru_cache(maxsize=1)
def _load_l1():
    return [YOLO(p) for p in L1_MODEL_PATHS]

@lru_cache(maxsize=1)
def _load_l2():
    return [YOLO(p) for p in L2_MODEL_PATHS]

@lru_cache(maxsize=1)
def _load_l3_early():
    return [YOLO(p) for p in L3_EARLY_MODEL_PATHS]

@lru_cache(maxsize=1)
def _load_l3_advanced():
    return [YOLO(p) for p in L3_ADVANCED_MODEL_PATHS]

# Ensemble prediction:
def _ensemble_predict(models: list, img: Image.Image):
    """
    Run a list of YOLO classification models on a single PIL image
    and return the ensemble-averaged prediction.
    
    Args:
        models : list of loaded ultralytics YOLO objects.
        img : PIL.Image (RGB) to classify.
    
    Returns:
        idx : (int) - index of the winning class.
        label : (str) - class name from the model's names dict.
        conf : (float) -averaged probability of the winning class.
    """
    all_probs, names_ref = [], None
    for model in models:
        result = model(img, verbose=False)[0] 
        all_probs.append(result.probs.data.cpu().numpy())
        if names_ref is None:
            names_ref = result.names

    avg_probs = np.mean(all_probs, axis=0)
    idx = int(np.argmax(avg_probs))
    label = names_ref[idx]
    conf = float(avg_probs[idx])
    return idx, label, conf