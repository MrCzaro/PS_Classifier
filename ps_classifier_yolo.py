from examples_config import *
from image_utils import annotate_image
import numpy as np
from functools import lru_cache
from PIL import Image
from ultralytics import YOLO

BINARY_POS_THRESHOLD = 0.5

BINARY_MODEL_PATHS = [
    "models/binary_yolo11l_aug_15_25.pt",
    "models/binary_yolov8l-cls_best_aug_11.pt",
]
STAGE_MODEL_PATHS = [
    "models/mutliclass_yolo11l_aug_15_25.pt",
    "models/multiclass_yolov8s-cls_best_aug_13.pt",
]

def _norm(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").strip()

@lru_cache(maxsize=1)
def load_yolo_models():
    binary = [YOLO(p) for p in BINARY_MODEL_PATHS]
    stage = [YOLO(p) for p in STAGE_MODEL_PATHS]
    return binary, stage

def _ensemble_predict(models, img: Image.Image):
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

def classify_image_ps(img_input):
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    else:
        img = Image.fromarray(img_input).astype(np.uint8)
    binary_models, stage_models = load_yolo_models()

    b_idx, b_label, b_conf = _ensemble_predict(binary_models, img)
    is_pressure = ("pressure" in _norm(b_label) and not _norm(b_label).startswith("not "))
    if not is_pressure or b_conf < BINARY_POS_THRESHOLD:
        annotated = annotate_image(img_input, b_label, b_conf, font_size=20)
        msg= f"No pressure sore detected ({b_conf:.2f} confidence) — stage model skipped"
        return (annotated or img), msg
    
    s_idx, s_label, s_conf = _ensemble_predict(stage_models, img)
    annotated = annotate_image(img_input, s_label, s_conf, font_size=20)
    msg = f"✅ Pressure sore detected\nStage: {s_label} \n({s_conf:.2f} confidence)"
    return (annotated or img), msg

