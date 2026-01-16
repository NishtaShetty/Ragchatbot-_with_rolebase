from .layoutlmv3_runner import run_layoutlmv3, is_flowchart_layoutlm

def detect_flowchart_with_layoutlm(img, all_words):
    """
    img       : OpenCV image
    all_words : OCR words from your pipeline
                [{ "text": str, "bbox": (x1,y1,x2,y2) }]
    """

    ocr_words = [
        {"text": w["text"], "bbox": w["bbox"]}
        for w in all_words
    ]

    words, boxes, preds = run_layoutlmv3(img, ocr_words)

    if not words:
        return False

    return is_flowchart_layoutlm(words)
