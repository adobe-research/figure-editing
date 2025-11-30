import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1_input, img2_input):
    """
    Compute SSIM between two images (file paths or NumPy arrays).
    Automatically resizes second image to match first.
    Returns: SSIM (float between -1 and 1)
    """

    # Load images if paths are given
    if isinstance(img1_input, str):
        img1 = cv2.imread(img1_input)
    else:
        img1 = img1_input

    if isinstance(img2_input, str):
        img2 = cv2.imread(img2_input)
    else:
        img2 = img2_input

    # Resize img2 to match img1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert both to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    score, _ = ssim(gray1, gray2, full=True)
    return score
