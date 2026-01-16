import torch
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# LOAD MODEL ONCE
# -------------------------------------------------
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False  # IMPORTANT: OCR already done
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base"
).to(DEVICE)

model.eval()

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def normalize_box(box, width, height):
    """
    Convert pixel bbox → LayoutLMv3 0-1000 format
    """
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


# -------------------------------------------------
# CORE INFERENCE
# -------------------------------------------------
def run_layoutlmv3(image_bgr, ocr_words):
    """
    image_bgr : OpenCV image (BGR)
    ocr_words : [
        {"text": str, "bbox": (x1,y1,x2,y2)}
    ]
    """

    # Convert image
    image_rgb = image_bgr[:, :, ::-1]
    image = Image.fromarray(image_rgb)
    width, height = image.size

    words, boxes = [], []

    for w in ocr_words:
        text = w["text"].strip()
        if not text:
            continue
        words.append(text)
        boxes.append(normalize_box(w["bbox"], width, height))

    if not words:
        return [], [], None

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).cpu().numpy()

    return words, boxes, predictions


# -------------------------------------------------
# FLOWCHART SIGNAL (PRACTICAL)
# -------------------------------------------------
def is_flowchart_layoutlm(words):
    """
    Heuristic over LayoutLMv3-aware tokens
    """
    decision_terms = {
        "yes", "no", "true", "false",
        "decision", "start", "end"
    }

    hits = sum(1 for w in words if w.lower() in decision_terms)

    return hits >= 2 and len(words) > 20
