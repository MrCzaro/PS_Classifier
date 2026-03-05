# Pressure Sore AI Classifier 🏥

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-00ADD8.svg)](https://github.com/ultralytics/ultralytics)
[![FastHTML](https://img.shields.io/badge/FastHTML-Latest-green.svg)](https://fastht.ml/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit)

Deep learning web application for automated pressure sore (pressure ulcer) detection and severity classification using ensemble neural networks and cascade architecture.



---

## 📸 Application Preview

### User Interface
<table>
  <tr>
    <td width="100%">
      <img src="docs/screenshots/dashboard.png" alt="Main Dashboard">
      <p align="center"><b>Interactive Dashboard</b><br/>Upload images or select from examples</p>
    </td>   
  </tr>
</table>
<table>
  <tr>
    <td width="50%">
      <img src="docs/screenshots/demo_ss_upload_2.png" alt="Upload Interface">
      <p align="center"><b>Drag & Drop Upload</b><br/>Seamless file handling with preview</p>
    </td>
    <td width="50%">
      <img src="docs/screenshots/demo_ss_upload_1.png" alt="Upload Interface">
      <p align="center"><b>Drag & Drop Upload</b><br/>Seamless file handling with preview</p>
    </td>
  </tr>
</table>

### Model Performance Examples
<table>
  <tr>
    <td width="33%">
      <img src="docs/screenshots/demo_ss_1.png" alt="Stage I PyTorch">
      <p align="center"><b>Stage I Detection PyTorch</b><br/>Early-stage pressure sore identification</p>
    </td>
    <td width="33%">
      <img src="docs/screenshots/demo_ss_2.png" alt="No pressure sore PyTorch">
      <p align="center"><b>Negative Classification PyTorch</b><br/>Accurate rejection of non-pressure sores</p>
    </td>
    <td width="33%">
      <img src="docs/screenshots/demo_ss_3.png" alt="Stage IV PyTorch">
      <p align="center"><b>Advanced Stage Classification PyTorch</b><br/>Deep tissue damage assessment</p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td width="33%">
      <img src="docs/screenshots/demo_ss_4.png" alt="Stage I YOLO">
      <p align="center"><b>Stage I Detection YOLO </b><br/>Early-stage pressure sore identification</p>
    </td>
    <td width="33%">
      <img src="docs/screenshots/demo_ss_5.png" alt="No pressure sore YOLO">
      <p align="center"><b>Negative Classification YOLO</b><br/>Accurate rejection of non-pressure sores</p>
    </td>
    <td width="33%">
      <img src="docs/screenshots/demo_ss_6.png" alt="Stage IV YOLO">
      <p align="center"><b>Advanced Stage Classification YOLO</b><br/>Deep tissue damage assessment</p>
    </td>
  </tr>
</table>

---

## 🎯 Project Overview

This project implements multiple deep learning pipeline strategies for automated pressure sore detection and classification, selectable at runtime via a backend dropdown. The system combines state-of-the-art computer vision models in ensemble architectures to achieve robust performance across three distinct backends.

### Medical Context

**Pressure sores (pressure ulcers)** are localized injuries to skin and underlying tissue, typically over bony prominences, caused by prolonged pressure. They are staged from I to IV based on severity:

- **Stage I**: Non-blanchable erythema (redness) of intact skin
- **Stage II**: Partial-thickness skin loss with exposed dermis
- **Stage III**: Full-thickness skin loss (fat visible)
- **Stage IV**: Full-thickness tissue loss (muscle/bone exposed)

Early detection and accurate staging are critical for treatment planning and preventing progression.

---
## ✨ Key Features

### 🧠 Three Selectable Backends

#### Backend 1 — Torchvision Cascade (PyTorch)
**Stage 1 — Binary Detection**: 5-model ensemble (ConvNeXt-Tiny, MaxViT-T, EfficientNet-B4, ResNet-50, Swin-V2-T)  
**Stage 2 — Multi-Class Staging**: 2-model ensemble (EfficientNet-B1, EfficientNet-V2-M)  
**Weights**: Available on request — see [Obtaining Model Weights](#-obtaining-model-weights)

#### Backend 2 — YOLO 2-Stage ← NEW
**Stage 1 — Binary Detection**: 3-model YOLO ensemble (YOLO11s, YOLOv8x, YOLO26x)  
**Stage 2 — Multi-Class Staging**: 3-model YOLO ensemble (YOLOv8n, YOLO26m, YOLOv8x)  
**Weights**: 👉 [MrCzaro/Pressure_sore_classifier_YOLO](https://huggingface.co/MrCzaro/Pressure_sore_classifier_YOLO)

#### Backend 3 — YOLO Cascade (3-Level Hierarchical)
**Level 1 — PS vs No-PS**: 2-model ensemble (YOLOv8x, YOLO26n)  
**Level 2 — Early vs Advanced**: 2-model ensemble (YOLOv8m, YOLO26x)  
**Level 3a — Stage I vs II**: 2-model ensemble (YOLOv8n, YOLO26x)  
**Level 3b — Stage III vs IV**: 2-model ensemble (YOLOv8x, YOLO26x)  
**Weights**: 👉 [MrCzaro/Pressure_sore_cascade_classifier_YOLO](https://huggingface.co/MrCzaro/Pressure_sore_cascade_classifier_YOLO)

---

## 🏗️ Architecture

### Backend 1 — Torchvision Cascade

```
Image → [Binary Ensemble: 5 models] → PS detected?
           NO → Return negative
           YES → [Staging Ensemble: 2 models] → Stage I/II/III/IV
```

### Backend 2 — YOLO 2-Stage

```
Image → [Binary YOLO Ensemble: YOLO11s + YOLOv8x + YOLO26x] → PS detected?
           NO → Return negative
           YES → [Multiclass YOLO Ensemble: YOLOv8n + YOLO26m + YOLOv8x] → Stage I/II/III/IV
```

### Backend 3 — YOLO Cascade (3-Level)

```
Image
  │
  ▼
[L1] PS vs No-PS  (YOLOv8x + YOLO26n)
  │ NO  → Return negative
  │ YES ↓
[L2] Early (I/II) vs Advanced (III/IV)  (YOLOv8m + YOLO26x)
  ├─ EARLY    → [L3a] Stage I vs Stage II   (YOLOv8n + YOLO26x)
  └─ ADVANCED → [L3b] Stage III vs Stage IV (YOLOv8x + YOLO26x)
```
---

## 🧪 Research & Development

### Experimental Approaches

#### ✅ Current Implementation 1 — Torchvision Cascade Ensemble
**Architecture**: Binary detection → Multi-class staging  
**Rationale**: Mimics clinical workflow — detect first, then stage  
**Pros**: High specificity, interpretable confidence scores  
**Challenges**: Stage III/IV differentiation remains difficult

#### ✅ Current Implementation 2 — YOLO 2-Stage
**Architecture**: Binary YOLO ensemble → Multiclass YOLO ensemble  
**Rationale**: Leverages YOLO's fast inference and classification head for both stages in a single framework  
**Binary stage**: Near-perfect detection (AUC up to 1.0000 across all 3 models)  
**Staging**: Macro F1 0.8514 ensemble — significant improvement over Torchvision baseline (0.74)

#### ✅ Current Implementation 3 — YOLO Cascade (3-Level)
**Architecture**: Hierarchical binary decisions — PS → Early/Advanced → Stage I/II or III/IV  
**Rationale**: Decomposes the hard 4-class problem into simpler binary boundaries  
**Joint F1**: 0.9256 (Early path), 0.8916 (Advanced path)  
**Key finding**: Isolating Stage III/IV from Stage I/II was the primary driver of improvement on the hardest classification pair

#### 🔬 Also Tested
- **DINOv2** (Meta AI self-supervised ViT): Competitive performance but high computational cost
- **YOLO object detection** (bounding box mode): Struggled with localization in medical images

---

---
## 📊 Model Performance

### Backend 1 — Torchvision Cascade

#### Stage 1 — Binary Classification

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ConvNeXt-Tiny    | 0.97 | 0.96 | 0.96 | 0.96 |
| MaxViT-T         | 0.98 | 0.97 | 0.98 | 0.98 |
| EfficientNet-B4  | 0.99 | 0.99 | 0.99 | 0.99 |
| ResNet-50        | 0.99 | 0.98 | 0.98 | 0.97 |
| Swin-V2-T        | 0.96 | 0.96 | 0.96 | 0.96 |
| **Ensemble**     | **0.98** | **0.97** | **0.97** | **0.97** |

#### Stage 2 — Multi-Class Staging

| Model | Accuracy | Macro F1 | Stage III/IV F1 |
|-------|----------|----------|-----------------|
| EfficientNet-B1   | 0.72 | 0.72 | 0.72 |
| EfficientNet-V2-M | 0.77 | 0.77 | 0.77 |
| **Ensemble**      | **0.74** | **0.74** | **0.74** |

---

### Backend 2 — YOLO 2-Stage

#### Stage 1 — Binary Detection (3-Model YOLO Ensemble)

| Model | Accuracy | Macro F1 | AUC-ROC |
|-------|----------|----------|---------|
| YOLO11s-cls  | 1.0000 | 1.0000 | 0.9998 |
| YOLOv8x-cls  | 1.0000 | 1.0000 | 1.0000 |
| YOLO26x-cls  | 1.0000 | 1.0000 | 1.0000 |
| **Ensemble** | **1.0000** | **1.0000** | **—** |

#### Stage 2 — Multi-Class Staging (3-Model YOLO Ensemble)

| Model | Accuracy | Macro F1 | Macro AUC-ROC (OvR) |
|-------|----------|----------|---------------------|
| YOLOv8n-cls  | 0.8571 | 0.8500 | 0.9486 |
| YOLO26m-cls  | 0.8571 | 0.8542 | 0.9306 |
| YOLOv8x-cls  | 0.8571 | 0.8501 | 0.9247 |
| **Ensemble** | **0.8571** | **0.8514** | **—** |

**Per-class AUC-ROC:**

| Stage | YOLOv8n | YOLO26m | YOLOv8x |
|-------|---------|---------|---------|
| Stage I   | 0.9913 | 0.9712 | 0.9934 |
| Stage II  | 0.9455 | 0.9261 | 0.9268 |
| Stage III | 0.8852 | 0.8866 | 0.8450 |
| Stage IV  | 0.9723 | 0.9386 | 0.9337 |

The 3-model YOLO ensemble achieves **0.8514 Macro F1** on staging — a significant improvement over the Torchvision baseline (0.74) using a simpler single-framework pipeline.

---

### Backend 3 — YOLO Cascade: Per-Level Performance

#### Level 1 — Binary: PS vs No-PS

| Model | Accuracy | Macro F1 | AUC-ROC |
|-------|----------|----------|---------|
| YOLOv8x-cls  | 1.0000 | 1.0000 | 1.0000 |
| YOLO26n-cls  | 1.0000 | 1.0000 | 0.9991 |
| **Ensemble** | **1.0000** | **1.0000** | **—** |

#### Level 2 — Binary: Early (Stage I/II) vs Advanced (Stage III/IV)

| Model | Accuracy | Macro F1 | AUC-ROC |
|-------|----------|----------|---------|
| YOLOv8m-cls  | 0.9615 | 0.9616 | 0.9948 |
| YOLO26x-cls  | 0.9615 | 0.9616 | 0.9969 |
| **Ensemble** | **0.9615** | **0.9616** | **—** |

#### Level 3a — Binary: Stage I vs Stage II *(Early branch)*

| Model | Accuracy | Macro F1 | AUC-ROC |
|-------|----------|----------|---------|
| YOLOv8n-cls  | 1.0000 | 1.0000 | 0.9501 |
| YOLO26x-cls  | 0.9286 | 0.9289 | 0.9521 |
| **Ensemble** | **0.9643** | **0.9644** | **—** |

#### Level 3b — Binary: Stage III vs Stage IV *(Advanced branch)*

| Model | Accuracy | Macro F1 | AUC-ROC |
|-------|----------|----------|---------|
| YOLOv8x-cls  | 0.9286 | 0.9289 | 0.9126 |
| YOLO26x-cls  | 0.9286 | 0.9289 | 0.8366 |
| **Ensemble** | **0.9286** | **0.9289** | **—** |

#### End-to-End Path Analysis

Joint F1 = product of ensemble F1 scores along each path (conservative lower bound):

| Cascade Path | L1 F1 | L2 F1 | L3 F1 | Joint F1 |
|---|---|---|---|---|
| PS → Early → **Stage I or II**      | 0.9981 | 0.9616 | 0.9644 | **0.9256** |
| PS → Advanced → **Stage III or IV** | 0.9981 | 0.9616 | 0.9289 | **0.8916** |

Both paths exceed the flat Torchvision staging baseline (0.74 macro F1).

---

## 🔑 Obtaining Model Weights

### Backend 2 — YOLO 2-Stage

👉 **[MrCzaro/Pressure_sore_classifier_YOLO](https://huggingface.co/MrCzaro/Pressure_sore_classifier_YOLO)**

```bash
git lfs install
git clone https://huggingface.co/MrCzaro/Pressure_sore_classifier_YOLO
```

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="MrCzaro/Pressure_sore_classifier_YOLO",
    filename="Binary PS or not PS YOLO8x.pt"
)
```

### Backend 3 — YOLO Cascade

👉 **[MrCzaro/Pressure_sore_cascade_classifier_YOLO](https://huggingface.co/MrCzaro/Pressure_sore_cascade_classifier_YOLO)**

```bash
git lfs install
git clone https://huggingface.co/MrCzaro/Pressure_sore_cascade_classifier_YOLO
```

### Backend 1 — Torchvision

Weights (~800MB) available on request:

```
📧 Email: cezary.tubacki@gmail.com
💬 Subject: "PS_Classifier Weights Request"
📝 Include: Your name, affiliation, and intended use case
```

---


## 🚀 Getting Started

### Prerequisites

```bash
Python 3.11+
PyTorch 2.0+ 
Ultralytics (for YOLO and YOLO Cascade backends)
CUDA-capable GPU (recommended, CPU supported)
8GB+ RAM
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MrCzaro/PS_Classifier.git
cd PS_Classifier
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLO weights**
```bash
# Backend 2
git lfs install
git clone https://huggingface.co/MrCzaro/Pressure_sore_classifier_YOLO models/yolo_2stage

# Backend 3
git clone https://huggingface.co/MrCzaro/Pressure_sore_cascade_classifier_YOLO models/yolo_cascade
```

5. **Update weight paths** in `ps_classifier_yolo.py` and `ps_classifier_yolo_cascade.py` (see Technical Stack section above)

6. **Run the application**
```bash
python main.py
# Open http://localhost:5001
```

---

### First-Time Setup

1. **Create account**: Navigate to signup page
2. **Test with examples**: Click any example image to classify
3. **Upload custom image**: Drag and drop your own pressure sore images
4. **Review predictions**: See annotated results with confidence scores

---
 

## 💻 Usage

### Web Interface

**Selecting a backend**:
Use the "Model backend" dropdown to choose between Torchvision, YOLO, and YOLO Cascade before classifying. The active backend is shown as a badge on every result.

**Example Classification**:
1. Select a backend from the dropdown
2. Click any example image → "Classify"
3. View annotated result with prediction + confidence

**Custom Upload**:
1. Select a backend from the dropdown
2. Drag image into upload zone (or click to browse)
3. Click "Classify Image"
4. Results display with backend badge and cascade path info

### Programmatic API

```python
# Two-stage backends
from ps_classifier import classify_image_ps        # Torchvision
from ps_classifier_yolo import classify_image_ps   # YOLO

annotated_img, message = classify_image_ps("path/to/image.jpg")
print(message)
# "✅ Pressure sore detected\nStage: stage III\n(0.87 confidence)"

# Three-level YOLO Cascade — full detail
from ps_classifier_yolo_cascade import classify_image_cascade, cascade_confidence

final_image, message, details = classify_image_cascade("path/to/image.jpg")
print(message)
# "✅ Pressure sore detected
#  Severity group: early (0.84)
#  Stage: stage II (0.79)"

print(details)
# {
#   "level_1": {"label": "pressure sore", "confidence": 0.97},
#   "level_2": {"label": "early",         "confidence": 0.84},
#   "level_3": {"label": "stage II",      "confidence": 0.79, "group": "Early"}
# }

# Joint confidence across all three levels
print(cascade_confidence(details))  # 0.97 × 0.84 × 0.79 ≈ 0.644

# Drop-in wrapper (same signature as original classify_image_ps)
from ps_classifier_yolo_cascade import classify_image_ps as classify_cascade
annotated_img, message = classify_cascade("path/to/image.jpg")
```

### Batch Processing

```python
import os
from pathlib import Path
from ps_classifier_yolo_cascade import classify_image_cascade, cascade_confidence

input_dir  = Path("images/to_classify")
output_dir = Path("results/annotated")
output_dir.mkdir(exist_ok=True)

for img_file in input_dir.glob("*.jpg"):
    result_img, message, details = classify_image_cascade(str(img_file))

    if result_img:
        result_img.save(output_dir / img_file.name)

        with open(output_dir / "predictions.txt", "a") as f:
            joint_conf = cascade_confidence(details)
            f.write(f"{img_file.name}: {message.strip()} | joint_conf={joint_conf}\n")
```

---

## 🛠️ Technical Stack

### Core Technologies

- **Backend**: [FastHTML](https://fastht.ml/) - Modern Python web framework
- **Frontend**: [MonsterUI](https://monsterui.org/) - Tailwind CSS + DaisyUI components
- **Deep Learning**:
  - **PyTorch** - Neural network framework
  - **Ultralytics YOLO** - Object detection/classification framework
- **Computer Vision**: 
  - TorchVision (pretrained models)
  - Albumentations (augmentation - PyTorch only)
  - PIL (image processing)
- **Database**: SQLite (user management)
- **Auth**: Bcrypt (password hashing)
- **Real-time**: HTMX (dynamic updates)

### Model Architecture Details

**Binary PyTorch Models** (5 models):
```python
binary_models_settings = {
    "ConvNeXt_Tiny": ["models/torch/final_model_ConvNeXt_Tiny_v2_model_head_mlp_binary_StepLR.pth", "mlp"],
    "MaxVit_T": ["models/torch/final_model_MaxVit_T_v2_model_head_linear_binary_CosineAnnealingLR.pth", "linear"],
    "EfficientNet_B4": ["models/torch/inal_model_EfficientNet_B4_v2_model_head_mlp_binary_StepLR.pth", "mlp"],
    "ResNet50": ["models/torch/final_model_ResNet50_v2_model_head_mlp_binary_CosineAnnealingLR.pth", "mlp"],
    "Swin_V2_T": ["models/torch/final_model_Swin_V2_T_v2_model_head_linear_binary_StepLR.pth", "linear"]
}
```

**Stage PyTorch Models** (2 models):
```python
stage_models_settings = {
    "EfficientNet_B1": ["models/torch/multiclass_EfficientNet_B1_Weights.IMAGENET1K_V2.pth", "linear"],
    "EfficientNet_V2_M": ["models/torch/multiclass_EfficientNet_V2_M_Weights.IMAGENET1K_V1.pth", "linear"]
}
```

**YOLO Models**:
```python
BINARY_MODEL_PATHS = [
    "models/yolo/Binary_YOLOv8x.pt",
    "models/yolo/Binary_YOLOv11s.pt",
    "models/yolo/Binary_YOLOv26x.pt"
]
STAGE_MODEL_PATHS = [
    "models/yolo/Multiclass_YOLOv8n.pt",
    "models/yolo/Multiclass_YOLOv8x.pt",
    "models/yolo/Multiclass_YOLOv26m.pt"
]
```

**YOLO Cascade** — 2 models per level, 4 levels, all `num_classes=1` binary:
```python
# Level 1 — PS vs No-PS 
L1_MODEL_PATHS = [
    "models/yolo_cascade/Level 1 Binary PS or not PS YOLO8x.pt",
    "models/yolo_cascade/Level 1 Binary PS or not PS YOLO26x.pt",
]

# Level 2 — Early (Stage I or II) vs Advanced (Stage III or IV)
L2_MODEL_PATHS = [
    "models/yolo_cascade/Level 2 Early vs Advanced YOLO26x.pt",
    "models/yolo_cascade/Level 2 Early vs Advanced YOLO8m.pt",
]

# Level 3a — Early group: Stage I vs Stage II
L3_EARLY_MODEL_PATHS = [
    "models/yolo_cascade/Level 3a Early YOLO8n.pt",
    "models/yolo_cascade/Level 3a Early YOLO26x.pt",
]

# Level 3b — Advanced group: Stage III vs Stage IV
L3_ADVANCED_MODEL_PATHS = [
    "models/yolo_cascade/Level 3b Advanced  YOLO8x.pt",
    "models/yolo_cascade/Level 3b Advanced  YOLO26x.pt",
]
```
---


## 🔒 Security & Privacy

**Medical Image Handling**:
- Images are NOT stored permanently (deleted after classification)
- No EXIF data is retained
- No patient identifiable information (PII) is collected

**User Authentication**:
- Bcrypt password hashing 
- Session-based auth with signed cookies
- SQL injection prevention via parameterized queries

**HIPAA Compliance Notes**:
- This is a demonstration/research tool
- NOT certified for clinical use without validation
- Users must ensure compliance when deploying

---



### Medical Disclaimer

**⚠️ IMPORTANT**:  
This software is provided **"as is"** for **research and educational purposes only**. It is:

- ❌ **NOT** a medical device
- ❌ **NOT** certified for clinical diagnosis
- ❌ **NOT** a substitute for professional medical judgment
- ❌ **NOT** validated in clinical trials

**Always consult licensed healthcare professionals for medical diagnosis and treatment.**

Use at your own risk. The author assumes no liability for decisions made based on this tool's output.

---

## 🙏 Acknowledgments

- **PyTorch Team**: For the incredible deep learning framework
- **Ultralytics**: For the YOLO framework and excellent documentation
- **FastHTML Creators**: For making web development enjoyable again
- **Medical Community**: For publicly available educational resources
- **Researchers**: Authors of pretrained models (ImageNet, COCO, etc.)
- **Open Source**: Standing on the shoulders of giants

---

## 📞 Contact & Support

**Author**: MrCzaro  
**GitHub**: [@MrCzaro](https://github.com/MrCzaro)
**Email**: cezary.tubacki@gmail.com  
**Project Link**: [PS_Classifier](https://github.com/MrCzaro/PS_Classifier)

**For**:
- Bug reports → [GitHub Issues](https://github.com/MrCzaro/PS_Classifier/issues)
- Feature requests → [GitHub Discussions](https://github.com/MrCzaro/PS_Classifier/discussions)


---




