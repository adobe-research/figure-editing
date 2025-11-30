import pytesseract
from PIL import Image
import cv2
import numpy as np
from difflib import SequenceMatcher

def compute_ocr_similarity(gt_path, test_path):
    """
    Compute OCR-based similarity between two images.
    Uses Tesseract OCR to extract text and compares via normalized edit distance.
    Returns: float in [0,1], where 1 means identical text.
    """

    def extract_text(img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image at {img_path} could not be read.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        pil_img = Image.fromarray(thresh)
        text = pytesseract.image_to_string(pil_img)
        return text.strip()

    try:
        gt_text = extract_text(gt_path)
        test_text = extract_text(test_path)
        similarity = SequenceMatcher(None, gt_text, test_text).ratio()
        return similarity
    except Exception as e:
        print(f"[Error] OCR similarity failed: {e}")
        return float("nan")
