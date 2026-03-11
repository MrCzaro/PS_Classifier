import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import numpy as np

from torchvision import models
from PIL import Image, UnidentifiedImageError
from albumentations.pytorch import ToTensorV2
from functools import lru_cache
from image_utils import annotate_image

# Error image path
ERROR_IMAGE_PATH = "static/error_picture.jpg"

# Confidence gates - Level 3 only
L3A_GATE: float = 0.65 # Stage I vs II (AUC ~0.99)
L3B_GATE: float = 0.65 # Stage III vs IV (AUC ~0.87)

# Model registry - architecture, checkpoint path, head type, dropout

# Level 1- PS vs No-PS (BCEWithLogitsLoss, sigmoid, num_classes=1)
L1_SETTINGS: dict[str, tuple[str, str, float]] = {
    "MaxVit_T": (
        "models/torch_cascade/Level 1 Binary PS or not PS MaxVit_T.pth",
        "linear", 0.2396,
    ),
    "ResNet50": (
        "models/torch_cascade/Level 1 Binary PS or not PS ResNet50.pth",
        "mlp", 0.5840,
    ),
}

# Level 2 — Early vs Advanced  (BCEWithLogitsLoss, sigmoid, num_classes=1)
L2_SETTINGS: dict[str, tuple[str, str, float]] = {
    "ConvNeXt_Base": (
        "models/torch_cascade/Level 2 Early vs Advanced ConvNeXt_Base.pth",
        "mlp", 0.1025,
    ),
    "EfficientNet_V2_L": (
        "models/torch_cascade/Level 2 Early vs Advanced EfficientNet_V2_L.pth",
        "linear", 0.3564,
    ),
}

# Level 3a — Stage I vs Stage II  (BCEWithLogitsLoss, sigmoid, num_classes=1)
L3A_SETTINGS: dict[str, tuple[str, str, float]] = {
    "EfficientNet_V2_L": (
        "models/torch_cascade/Level 3a Early EfficientNet_V2_L.pth",
        "mlp", 0.1949,
    ),
    "ConvNeXt_Tiny": (
        "models/torch_cascade/Level 3a Early ConvNeXt_Tiny.pth",
        "linear", 0.1601,
    ),
}

# Level 3b — Stage III vs Stage IV  (CrossEntropyLoss, softmax+argmax, WrappedModel)
L3B_SETTINGS: dict[str, tuple[str, str, float]] = {
    "ConvNeXt_Large": (
        "models/torch_cascade/Level 3b Advanced ConvNeXt_Large.pth",
        "multi_stage_head", 0.6594,
    ),
    "ViT_B_16": (
        "models/torch_cascade/Level 3b Advanced ViT_B_16.pth",
        "mlp", 0.5445,
    ),
}

# Label maps
L1_LABELS = ["not pressure sore", "pressure sore"]
L2_LABELS = ["early", "advanced"]
L3A_LABELS = ["Stage I", "Stage II"]
L3B_LABELS = ["Stage III", "Stage IV"]



# Head definitions (only needed for WrappedModel - L3B)
class MultiStageHead(nn.Module):
    """
    Two-layer classification head with dropout and batch normalization.

    Used by ConvNeXt_Large in the Level 3b stage classifier. The head
    reduces feature dimensionality and outputs logits for two classes.

    Args:
        in_features (int): Size of backbone feature embeddings.
        num_classes (int): Number of output classes (default=2).
        dropout (float): Dropout probability.

    Forward:
        x (Tensor): Feature embeddings [B, in_features].

    Returns:
        Tensor: Class logits [B, num_classes].
    """
    def __init__(self, in_features: int, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.bn = nn.BatchNorm1d(in_features // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features // 2, num_classes)

    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        return self.fc2(self.relu(self.bn(self.fc1(self.dropout(x)))))
    

class WrappedModel(nn.Module):
    """
    Wrapper combining a backbone feature extractor and a classification head.

    The backbone produces feature embeddings which are passed to the head
    to generate logits. Used mainly for Level 3b models where the head
    operates on extracted features rather than being attached directly
    to the backbone classifier.

    Args:
        backbone (nn.Module): Feature extractor returning embeddings.
        head (nn.Module): Classification head applied to the embeddings.

    Forward:
        x (Tensor): Input images [B, C, H, W]
        labels (Tensor, optional): Used only by margin-based heads.

    Returns:
        Tensor: Output logits.
    """
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        feats = self.backbone(x)
        # Standard heads (linear/mlp/multi_stage_head) ignore labels
        # margin heads would apply margin only during training — but L3b
        # models are not margin heads, so labels is always ignored here.
        return self.head(feats)



# Architecture helpers
TV_FN: dict[str, callable] = {
    "EfficientNet_B0" : models.efficientnet_b0,
    "EfficientNet_B1" : models.efficientnet_b1,
    "EfficientNet_B4" : models.efficientnet_b4,
    "EfficientNet_B7" : models.efficientnet_b7,
    "EfficientNet_V2_L" : models.efficientnet_v2_l,
    "ViT_B_16" : models.vit_b_16,
    "MaxVit_T" : models.maxvit_t,
    "Wide_ResNet50_2" : models.wide_resnet50_2,
    "ResNet50" : models.resnet50,
    "ResNet152" : models.resnet152,
    "Swin_V2_S" : models.swin_v2_s,
    "Swin_V2_T" : models.swin_v2_t,
    "ConvNeXt_Tiny" : models.convnext_tiny,
    "ConvNeXt_Base" : models.convnext_base,
    "ConvNeXt_Large" : models.convnext_large,
    "RegNet_Y_8GF" : models.regnet_y_8gf,
    "RegNet_Y_16GF" : models.regnet_y_16gf,
}

def get_in_features(arch: str, model: nn.Module) -> int:
    """Read in_feature from the backbone's native classifier slot."""
    if arch.startswith("ViT"): return model.heads.head.in_features
    if arch.startswith("MaxVit"): return model.classifier[5].in_features
    if arch.startswith("Conv"): return model.classifier[2].in_features
    if arch.startswith("Swin"): return model.head.in_features
    if arch.startswith("RegNet"): return model.fc.in_features
    if hasattr(model, "fc"): return model.fc.in_features
    return model.classifier[1].in_features # EfficientNet family

def load_weights(model: nn.Module, path: str) -> nn.Module:
    """
    Load model weights from a checkpoint file.

    Args:
        model : PyTorch model instance to receive the weights.
        path  : Path to the checkpoint file (.pth / .pt).

    Returns:
        The model with loaded weights, set to eval() mode and frozen parameters.
    """
    # Load checkpoint
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    # Remove DataParallel prefix
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    
    # Load weights and set model to inference mode
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Disable grad
    for param in model.parameters():
        param.requires_grad_(False)
    return model

# Model builders

def build_standard(arch: str, path: str, head: str = "linear", dropout: float = 0.3) -> nn.Module:
    """
    L1 / L2 / L3a pattern — identical to build_model() in ps_classifier.py.

    Head attached directly onto the backbone.

    Output:
        [B, 1] logit → sigmoid at inference → [neg_prob, pos_prob]

    Supported head_type:
        "linear" | "mlp"
    """
    # Build backbone
    backbone = TV_FN[arch](weights=None)
    in_feat = get_in_features(arch, backbone)
    
    if head == "linear":
        head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 1)
        )

    elif head == "mlp":
        head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, in_feat // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_feat // 2, 1)
        )
    else:
        raise ValueError(f"build_standard: unsupported head type `{head}`")
    
    # Attach head
    if arch.startswith("ViT"): backbone.heads.head = head
    elif arch.startswith("MaxVit"): backbone.classifier[5] = head
    elif arch.startswith("Conv"): backbone.classifier[2] = head
    elif arch.startswith("Swin"): backbone.head = head
    elif arch.startswith("RegNet"): backbone.fc = head
    elif hasattr(backbone, "fc"): backbone.fc = head
    else: backbone.classifier = head

    return load_weights(backbone, path)


def build_wrapped(
    arch: str, 
    path: str, 
    head: str = "mlp", 
    dropout: float = 0.3, 
    num_classes: int =2) -> WrappedModel:
    """
    L3b pattern — WrappedModel (CrossEntropyLoss).

    Backbone classifier stripped to Identity; head receives [B, in_features].

    Output:
        logits [B, num_classes] → softmax + argmax at inference

    Supported head_type:
        "linear" | "mlp" | "multi_stage_head"
    """
    if arch not in TV_FN:
        raise ValueError(f"Unknown architecture: {arch}")
    # Build backbone
    backbone = TV_FN[arch](weights=None)
    in_feat = get_in_features(arch, backbone)

    # Strip original classifier 
    if arch.startswith("ViT"):
        backbone.heads.head = nn.Identity()
    elif arch.startswith("MaxVit"):
        backbone.classifier[5] = nn.Identity()
    elif arch.startswith("Conv"):
        backbone.classifier[2] = nn.Identity()
    elif arch.startswith("Swin"):
        backbone.head = nn.Identity()
    elif arch.startswith("RegNet"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    else:
        backbone.classifier = nn.Identity()

    # Build head
    if head == "linear":
        head : nn.Module = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, num_classes)
        )
    elif head == "mlp":
        hidden = in_feat // 2
        head : nn.Module = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    elif head == "multi_stage_head":
        head = MultiStageHead(in_feat, num_classes, dropout)
    else:
        raise ValueError(f"build_wrapped: unsupported head_type '{head}'")

    # Wrap model
    model = WrappedModel(backbone, head)

    # Load weights
    return load_weights(model, path)

# Lazy loading - one cached list per cascade level
@lru_cache(maxsize=1)
def _load_l1() -> list[nn.Module]:
    return[build_standard(arch, model_path, head, dropout) for arch, (model_path, head, dropout) in L1_SETTINGS.items()]


@lru_cache(maxsize=1)
def _load_l2() -> list[nn.Module]:
    return[build_standard(arch, model_path, head, dropout) for arch, (model_path, head, dropout) in L2_SETTINGS.items()]

@lru_cache(maxsize=1)
def _load_l3a() -> list[nn.Module]:
    return[build_standard(arch, model_path, head, dropout) for arch, (model_path, head, dropout) in L3A_SETTINGS.items()]

@lru_cache(maxsize=1)
def _load_l3b() -> list[nn.Module]:
    return[build_wrapped(arch, model_path, head, dropout) for arch, (model_path, head, dropout) in L3B_SETTINGS.items()]

# Ensemble utilities
def ensemble_standard(models: list[nn.Module], tensor: torch.Tensor) -> np.ndarray:
    """
    BCEWithLogitsLoss ensemble (L1 / L2 / L3a).
    Returns averaged [p_neg, p_pos] array.
    """
    all_probs = []
    for model in models:
        with torch.inference_mode():
            logit = model(tensor).squeeze() # scalar
            pos = float(torch.sigmoid(logit))
        all_probs.append([1.0 - pos, pos])
    return np.mean(all_probs, axis=0) # shape[2]

def ensemble_wrapped(models: list[nn.Module], tensor: torch.Tensor) -> np.ndarray:
    """
    CrossEntropyLoss ensemble (L3b / WrappedModel).
    Returns averaged softmax probability array [num_classes].
    """
    all_probs = []
    for model in models:
        with torch.inference_mode():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
        all_probs.append(probs.cpu().numpy())
    return np.mean(all_probs, axis=0)

# 3 Level Cascade Classifier
def classify_image_cascade(img_input) -> tuple[Image.Image | None, str, dict]:
    """
    Full 3-level torchvision cascade with Level-3 confidence gating.
    
    Args:
        img_input: file path (str), PIL.Image, or numpy array.
        
    Returns:
        final_image : annotated PIL.Image (None only on load failure).
        message : human readable classfication summary.
        details : per level structured results.
    
    """
    # Load image
    try:
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
        elif hasattr(img_input, "__array__"):
            img = Image.fromarray(np.array(img_input).astype(np.uint8)).convert("RGB")
        else:
            return None, "Invalid input - provide a path, PIL.Image, or array.", {}
    except (UnidentifiedImageError, FileNotFoundError) as e:
        return None, f"Could not load image: {e}", {}
    
    details: dict = {}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Define and apply transformations for PyTorch models
    picture_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean,
                    std=std),
        ToTensorV2(),
    ])
    # Convert PIL image to NumPy array for Albumentations
    image_np = np.array(img)
    img_transformed = picture_transforms(image=image_np)["image"].unsqueeze(0)

    # Level 1: PS vs No PS
    l1_probs = ensemble_standard(_load_l1(), img_transformed)
    l1_idx = int(np.argmax(l1_probs))
    l1_label = L1_LABELS[l1_idx]
    l1_conf = float(l1_probs[l1_idx])
    details["level_1"] = {"label": l1_label, "confidence" : l1_conf}

    if l1_idx == 0: 
        msg = f"❌ No pressure sore detected ({l1_conf:.2f} confidence)"
        return annotate_image(img, l1_label, l1_conf), msg, details
    
    # Level 2: Early (I/II) vs Advanced (III/IV)
    try: 
        l2_probs = ensemble_standard(_load_l2(), img_transformed)
        l2_idx = int(np.argmax(l2_probs))
        l2_label = L2_LABELS[l2_idx]
        l2_conf = float(l2_probs[l2_idx])
        details["level_2"] = {"label": l2_label, "confidence" : l2_conf}
    except Exception as e:
        try:
            error_img = Image.open(ERROR_IMAGE_PATH).convert("RGB")
        except:
            error_img = Image.new("RGB", (224, 224), color=(180, 0, 0))
        msg = f"Error during Level-2 classification {e}"
        return error_img, msg, details
    
    is_early = (l2_idx == 0)

    # Level 3: Stage with confidence gate
    try:
        if is_early:
            # L3a - direct-attachment, BCE/sigmoid (same as L1/L2)
            l3_probs = ensemble_standard(_load_l3a(), img_transformed)
            l3_labels = L3A_LABELS
            group = "Early"
            gate = L3A_GATE
        else:
            # L3b - WrappedModel, CrossEntropy/softmax+argmax
            l3_probs = ensemble_wrapped(_load_l3b(), img_transformed)
            l3_labels = L3B_LABELS
            group = "Advanced"
            gate = L3B_GATE
        
        l3_idx = int(np.argmax(l3_probs))
        l3_label = l3_labels[l3_idx]
        l3_conf  = float(l3_probs[l3_idx])
        gated = l3_conf < gate

        details["level_3"] = {
            "label" : l3_label,
            "confidence" : l3_conf,
            "group" : group,
            "gated" : gated
        }

    except Exception as e:
        try:
            error_img = Image.open(ERROR_IMAGE_PATH).convert("RGB")
        except:
            error_img = Image.new("RGB", (224, 224), color=(180, 0, 0))
        msg = f"Error during Level-3 classification {e}"
        return error_img, msg, details

    # Build output
    cascade_path = (
        f"L1({l1_conf:.2f}) → {l2_label}({l2_conf:.2f}) → {l3_label}({l3_conf:.2f})"
        + (" ⚠" if gated else "")
    )

    if gated:
        msg = (
            f"✅ Pressure sore detected\n"
            f"Severity group: {l2_label} ({l2_conf:.2f})\n"
            f"Stage: {l3_label} ({l3_conf:.2f})\n"
            f"⚠  Low confidence at Level 3 ({l3_conf:.2f} < {gate:.2f}) - clinical review recommended"
        )
    else:
        msg = (
            f"✅ Pressure sore detected\n"
            f"Severity: {l2_label} ({l2_conf:.2f})\n"
            f"Stage: {l3_label} ({l3_conf:.2f})"
        )

    final_image = annotate_image(img, l3_label, l3_conf, cascade_info=cascade_path, uncertain=gated)
    return final_image, msg, details 

# Joint confidence utility
def cascade_confidence(details: dict) -> float:
    """
    Product of per-level coinfidences -> single joint score.
    Example: L1=0.99 * L2=0.95 * L3=0.79 = 0743
    """
    score = 1.0
    for level in ("level_1", "level_2", "level_3"):
        if level in details:
            score *= details[level]["confidence"]
    return round(score, 4)

# Wraper
def classify_image_ps(img_input) -> tuple[Image.Image, None, str]:
    """Classify an image wraper using a 3-level cascade of PyTorch models."""
    final_image, message, _ = classify_image_cascade(img_input)
    return final_image, message