import cv2
import numpy as np
import os

def compute_psnr(img1_input, img2_input, max_val=255.0):
    """
    Compute PSNR between two images (file paths or NumPy arrays).
    Automatically resizes img2 to match img1.
    Returns PSNR (float), or inf if identical.
    """

    if isinstance(img1_input, str):
        img1 = cv2.imread(img1_input)
    else:
        img1 = img1_input

    if isinstance(img2_input, str):
        img2 = cv2.imread(img2_input)
    else:
        img2 = img2_input

    if img1 is None or img2 is None:
        raise ValueError("Image load failed.")

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse)
