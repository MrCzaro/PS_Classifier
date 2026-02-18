from PIL import Image, ImageDraw, ImageFont
import numpy as np

def annotate_image(img, label, confidence, font_size=30):
    """
    Anotates an image with predicted label and confidence.
    
    Args:
        img (PIL.Image): Input image (can be a PIL object or path to image file).
        label (str) : Predicted label name.
        confidence (float) : Model confidence value (0-1 or %).
        font_size (int) : Font size for annotation text.

    Supports:
        - PIL.Image
        - image path (string)
        - numpy array

    Returns:
        PIL.Image.Image (annotated)

    """
    # print(f"Input type to annotate_image: {type(img)}")
    
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
    except IOError:
        font = ImageFont.load_default()

    # Define text
    text_label = f"Predicted: {label.replace('_', ' ')}"
    text_confidence = f"Confidence: {confidence:.2f}"

    # Position text at top-left corner
    x,y = 10, 10
    draw.text((x,y), text_label, font=font, fill="yellow")
    draw.text((x,y + font_size + 5), text_confidence, font=font, fill="yellow")
    
    return annotated