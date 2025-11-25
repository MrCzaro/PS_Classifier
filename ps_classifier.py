import torch, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from pathlib import Path

def build_model(model_name, path_to_weights, num_classes=1, dropout=0.5, head_type="linear"):
    # Select architecture
    model = {
        "EfficientNet_B0": models.efficientnet_b0,
        "EfficientNet_B1" : models.efficientnet_b1,
        "EfficientNet_B3": models.efficientnet_b3,
        "EfficientNet_B4": models.efficientnet_b4,
        "EfficientNet_V2_M": models.efficientnet_v2_m,
        "ViT_B_16": models.vit_b_16,
        "MaxVit_T" : models.maxvit_t,
        "Wide_ResNet50_2" : models.wide_resnet50_2,
        "ResNet50" :  models.resnet50,
        "ResNet152" : models.resnet152,
        "Swin_V2_S" : models.swin_v2_s,
        "Swin_V2_T" : models.swin_v2_t,
        "ConvNeXt_Tiny" : models.convnext_tiny
    }[model_name](weights=None)

    # Extract in_features based on model type
    if model_name.startswith("ViT"):
        in_features = model.heads.head.in_features
    elif model_name.startswith("Max"):
        in_features = model.classifier[5].in_features
    elif model_name.startswith("Conv"):
        in_features = model.classifier[2].in_features
    elif model_name.startswith("Swin"):
        in_features = model.head.in_features
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
    else: # EfficientNet
        in_features = model.classifier[1].in_features

    # Build head
    if head_type == "linear":
        head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif head_type == "mlp":
        head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, num_classes)
        )
    # Assigne head back to model 
    if model_name.startswith("ViT"):
        model.heads.head = head
    elif model_name.startswith("Max"):
        model.classifier[5] = head
    elif model_name.startswith("Conv"):
        model.classifier[2] = head
    elif model_name.startswith("Swin"):
        model.head = head
    elif hasattr(model, "fc"):
        model.fc = head
    else: # EfficientNet
        model.classifier = head
    # Load weights 
    trained_weights_dict = torch.load(path_to_weights,  map_location=torch.device('cpu'))
    new_state_dict = {k: v for k, v in trained_weights_dict.items() if k in model.state_dict()}
    model.load_state_dict(new_state_dict)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def ensemble_predict(models, img, labels, binary=False):
    """
    Runs ensemble prediction over a list of PyTorch Vision models.
    Assumes models are already in .eval() model and have requires_grad=False
    """

    all_probs = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    IMG_SIZE = 224
    
    # Define and apply transformations for PyTorch models
    picture_transforms = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
    # Convert PIL image to NumPy array for Albumentations
    image_np = np.array(img)
    # Apply the transformations
    img_transformed = picture_transforms(image=image_np)["image"]

    for model in models:
        with torch.inference_mode():
                outputs = model(img_transformed.unsqueeze(0)).squeeze()
                if binary == True:
                    prob_positive = torch.sigmoid(outputs)
                    prob_negative = 1 - prob_positive   
                    probs = torch.cat((prob_negative.unsqueeze(0), prob_positive.unsqueeze(0))).squeeze()
                else:
                    probs = F.softmax(outputs, dim=0 if outputs.dim() == 1 else 1)
                    probs = probs.squeeze()
                all_probs.append(probs.cpu().numpy())

    avg_probs = np.mean(all_probs, axis=0)
    idx = int(np.argmax(avg_probs))
    conf = float(avg_probs[idx])
    final_label = labels[idx]
    return idx, final_label, conf

def annotate_image(img, label, confidence, font_size=20):
    """
    Anotates an image with predicted label and confidence.
    
    Args:
        img (PIL.Image): Input image (can be a PIL object or path to image file).
        label (str) : Predicted label name.
        confidence (float) : Model confidence value (0-1 or %).
        font_size (int) : Font size for annotation text.

    Returns:
        PIL.Image: Annotaged image

    """
    print(f"Input type to annotate_image: {type(img)}")
    try:
        # Open an image
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8)).convert("RGB") 
        else:
            print("Invalid input type. Please provide a path (string) or a NumPy array.")
            return None
    except(UnidentifiedImageError, FileNotFoundError) as e:
        print(f"Error opening image: {e}.\n Please provide a valide image path or array.")
        return None

    # Create drawing object
    draw = ImageDraw.Draw(img)
    
    # Load default font (or specify TTF ) 
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Define text
    text_label = f"Predicted label: {label}"
    text_confidence = f"Confidence: {confidence:.2f}"

    # Position text at top-left corner
    x,y = 10, 10
    draw.text((x,y), text_label, font=font, fill="white")
    draw.text((x,y + font_size + 5), text_confidence, font=font, fill="white")
    
    return img

def classify_image_ps(img_input):
    """
    1) Run binary classifier
    2) If positive (by index) and above treshold -> run stage classifier
    3) Else - > skip stage model.
    """

    try:
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            img = Image.fromarray(img_input.astype(np.uint8)).convert("RGB")
        else:
            print("Invalid input type. Please provide a path (string) or a NumPy array.")
            return None, "Invalid image path"
    except (UnidentifiedImageError, FileNotFoundError) as e:
        print(f"Error opening image: {e}.\n Please provide a valide image path or array.")
        return None, "Image could not be loaded"

    # Binary ensemble
    b_idx, b_label, b_conf = ensemble_predict(binary_models, img, binary_labels, binary=True)
    print(f"DEBUG Binary Ensemble:\n1. b_idx :  {b_idx}\n2. b_label: {b_label}\n3. b_conf: {b_conf}.")

    # Decide pressure sore
    is_pressure = ("pressure" in b_label and not b_label.startswith("not"))
    print(f"IS pressure sore? {is_pressure}")
    if not is_pressure:
        message = f"No pressure sore detected ({b_conf:.2f} confidence) -- stage model skipped"
        final_image = annotate_image(img=img_input, label=b_label, confidence=b_conf, font_size=20)
    else:
        print("Running stage model...")
        # Stage ensemble 
        try:
            s_idx, s_label, s_conf = ensemble_predict(stage_models, img, multiclass_labels, binary=False)
            print(f"DEBUG Stage Ensemble:\n1. s_idx: {s_idx}\n2. s_label: {s_label}\n3. s_conf: {s_conf}.")
            message = f"✅ Pressure sore detected\nStage: {s_label} \n({s_conf:.2f} confidence)"
            final_image = annotate_image(img=img_input, label=s_label, confidence=s_conf, font_size=20)
        except:
            final_image = Image.open("pic/error_picture.jpg").convert("RGB")
            message = "Error during stage classification. Please check the model or input image."
    return final_image, message

binary_models_settings = {
    "ConvNeXt_Tiny" : ["models/final_model_ConvNeXt_Tiny_v2_model_head_mlp_binary_StepLR.pth", "mlp"],
    "MaxVit_T" : ["models/final_model_MaxVit_T_v2_model_head_linear_binary_CosineAnnealingLR.pth", "linear"],
    "EfficientNet_B4" : ["models/final_model_EfficientNet_B4_v2_model_head_mlp_binary_StepLR.pth", "mlp"],
    "ResNet50" : ["models/final_model_ResNet50_v2_model_head_mlp_binary_CosineAnnealingLR.pth", "mlp"],
    "Swin_V2_T" : ["models/final_model_Swin_V2_T_v2_model_head_linear_binary_StepLR.pth", "linear"]

}


stage_models_settings = {
    "EfficientNet_B1" : ["models/multiclass_EfficientNet_B1_Weights.IMAGENET1K_V2.pth", "linear" ],
    "EfficientNet_V2_M" : ["models/multiclass_EfficientNet_V2_M_Weights.IMAGENET1K_V1.pth", "linear"],

}

# Load PyTorch models
binary_models = [build_model(model_name=name, path_to_weights=values[0], head_type=values[1], num_classes=1) for name, values in binary_models_settings.items()]
stage_models = [build_model(model_name=name, path_to_weights=values[0], head_type=values[1], num_classes=4) for name, values in stage_models_settings.items()]

# Labels
binary_labels = ["not pressure_sore", "pressure sore"]
multiclass_labels = ["stage I", "stage II", "stage III", "stage IV"]

# Examples preperation
BASE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = BASE_DIR / "static/"
pressure_examples = [
    os.path.join(EXAMPLES_DIR, f"pressure_{i}.jpg") for i in range(1, 4)
]
no_pressure_examples = [
    os.path.join(EXAMPLES_DIR, f"no_pressure_{i}.jpg") for i in range(1, 4)
]
examples = pressure_examples + no_pressure_examples
