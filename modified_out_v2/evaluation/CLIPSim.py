import torch
import clip
from PIL import Image
import os

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_similarity(img1_path, img2_path):
    """
    Compute CLIP image-to-image cosine similarity.
    Returns a float in [-1, 1] (higher is more similar).
    """

    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        raise FileNotFoundError("Image path does not exist")

    try:
        img1 = preprocess(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)
        img2 = preprocess(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            feat1 = clip_model.encode_image(img1)
            feat2 = clip_model.encode_image(img2)
            feat1 /= feat1.norm(dim=-1, keepdim=True)
            feat2 /= feat2.norm(dim=-1, keepdim=True)

        similarity = (feat1 @ feat2.T).item()
        return similarity
    except Exception as e:
        raise RuntimeError(f"CLIP similarity failed: {e}")
