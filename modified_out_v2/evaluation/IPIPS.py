import torch
import lpips
import cv2
import numpy as np

# Initialize once
lpips_model = lpips.LPIPS(net='alex')  # or 'vgg', 'squeeze'

def load_and_preprocess(path_or_array, size=None):
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array)
    else:
        img = path_or_array.copy()

    if img is None:
        raise ValueError("Failed to load image")

    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalize to [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return 2 * img - 1  # Normalize to [-1, 1] for LPIPS

def compute_lpips(img1_input, img2_input):
    # Load first image
    if isinstance(img1_input, str):
        img1_cv = cv2.imread(img1_input)
    else:
        img1_cv = img1_input

    if isinstance(img2_input, str):
        img2_cv = cv2.imread(img2_input)
    else:
        img2_cv = img2_input

    if img1_cv is None or img2_cv is None:
        raise ValueError("Failed to load image")

    h, w = img1_cv.shape[:2]
    img2_cv = cv2.resize(img2_cv, (w, h))  # match size

    img1_tensor = load_and_preprocess(img1_cv)
    img2_tensor = load_and_preprocess(img2_cv)

    with torch.no_grad():
        dist = lpips_model(img1_tensor, img2_tensor)
    return dist.item()
