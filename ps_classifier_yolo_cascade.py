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


# 3 Level Cascade Classifier
def classify_image_cascade(img_input):
    """
    Full 3-level YOLO cascade inference.
        
    Args:
        img_input : file path (str) or PIL.Image.

    Returns:
        final_image (PIL.Image) : annotated result.
        message (str) : human readable summary.
        details (dict) : per-level structured results for logging/UI.
    """

    # Load image
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        img = Image.fromarray(np.array(img_input).astype(np.uint8)).convert("RGB")

    details = {}

    # Level 1: PS vs No PS
    l1_idx, l1_label, l1_conf = _ensemble_predict(_load_l1(), img)
    details["level_1"] = {"label": l1_label, "confidence" : l1_conf}

    is_ps = ("pressure" in _norm(l1_label) and not _norm(l1_label).startswith("not") and l1_conf >= BINARY_THRESHOLD) 

    if not is_ps:
        message = f"❌ No pressure sore detected ({l1_conf:.2f} confidence)"
        annotated = annotate_image(img, l1_label, l1_conf, font_size=20)
        return (annotated or img), message, details
    
    # Level 2: Early (I/II) vs Advanced (III/IV)
    try:
        l2_idx, l2_label, l2_conf = _ensemble_predict(_load_l2(), img)
        details["level_2"] = {"label": l2_label, "confidence" : l2_conf}
    except Exception as e:
        message = f"Error at severity triage (Level 2): {e}"
        return img, message, details
    
    is_early = "early" in _norm(l2_label)

    # Level 3: Fine-grained within the chosen group
    try:
        if is_early:
            l3_idx, l3_label, l3_conf = _ensemble_predict(_load_l3_early(), img)
            details["level_3"] = {"label" : l3_label, "confidence" : l3_conf}
            group_path = "Early"
        else: 
            l3_idx, l3_label, l3_conf = _ensemble_predict(_load_l3_advanced(), img)
            group_path = "Advanced"
        details["level_3"] = {"label": l3_label, "confidence" : l3_conf, "group" : group_path}
    except Exception as e:
        message = f"Error at stage classification (Level 3): {e}"
        return img, message, details
    
    # Build output
    message = (
        f"✅ Pressure sore detected\n"
        f"Severity group: {l2_label} ({l2_conf:.2f})\n"
        f"Stage: {l3_label} ({l3_conf:.2f})"
    )
    annotated = annotate_image(img, l3_label, l3_conf, font_size=20)
    return (annotated or img), message, details

def cascade_confidence(details: dict) -> float:
    """
    Multiply per-level confidences into a single joint confidence score.
    
    Example: L1=0.97, L2=0.84, L3=0.79 -> 0.97 x 0.84 x 0.79 ~0.644
    
    Usefyl for flagging low-certainty predictions for review.
    """
    score = 1.0
    for level in ("level_1", "level_2", "level_3"):
        if level in details:
            score *= details[level]["confidence"]
    return round(score, 4)

# Wraper
def classify_image_ps(img_input):
    """Thin wrapper preserving the (final_image, message) return contract."""
    final_image, message, _ = classify_image_cascade(img_input)
    return final_image, message