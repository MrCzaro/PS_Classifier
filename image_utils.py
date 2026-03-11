from PIL import Image, ImageDraw, ImageFont
import numpy as np

def annotate_image(
    img: Image.Image, 
    label: str, 
    confidence: float, 
    cascade_info: str = "", 
    font_size: int=20, 
    uncertain: bool = False) -> Image.Image:
    """
    Anotates an image with predicted label and confidence.
    
    Args:
        img (PIL.Image): Input image (can be a PIL object or path to image file).
        label (str) : Predicted label name.
        confidence (float) : Model confidence value (0-1 or %).
        cascade_info (str) : Optional additional info from cascade levels(Torch). 
        font_size (int) : Font size for annotation text.
        uncertain (bool) : If True, annotate with "Uncertain Prediction" instead of label/confidence.

    Supports:
        - PIL.Image
        - image path (string)
        - numpy array

    Returns:
        PIL.Image.Image (annotated)

    """
    
    try:
        if isinstance(img, Image.Image):
            annotated = img.copy() 
        elif isinstance(img, str):
            annotated = Image.open(img).convert("RGB")
    
        elif isinstance(img, np.ndarray):
            annotated = Image.fromarray(img.astype(np.uint8)).convert("RGB") 
        else:
            return None
    except Exception as e:
        # print(f"Error opening image: {e}.\n Please provide a valide image path or array.")
        return None

    # Create drawing object
    draw = ImageDraw.Draw(annotated)

    # Load default font (or specify TTF ) 
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        small = ImageFont.truetype("arial.ttf", max(font_size - 6, 12))
    except IOError:
        font = ImageFont.load_default()
        small = font

    colour = "orange" if uncertain else "yellow"
    x,y = 10, 10
    step = font_size + 45

    draw.text((x,y), f"Stage: {label}", font=font, fill=colour)
    draw.text((x,y + step), f"Confidence: {confidence:.2f}", font=font, fill=colour)
    if cascade_info:
        draw.text((x,y + step *2), cascade_info, font=small, fill="cyan")
    if uncertain:
        draw.text((x,y + step * 3), "Uncertain Prediction", font=small, fill="red")

    return annotated