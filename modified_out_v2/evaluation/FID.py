from cleanfid import fid
import os

def compute_fid(gt_dir, gen_dir):
    """
    Compute FID between two directories using clean-fid.
    Automatically checks validity of input paths.
    Returns FID score (float).
    """
    if not (os.path.exists(gt_dir) and os.path.exists(gen_dir)):
        raise FileNotFoundError(f"Missing path: {gt_dir} or {gen_dir}")

    try:
        score = fid.compute_fid(gt_dir, gen_dir)
    except Exception as e:
        raise RuntimeError(f"FID computation failed: {e}")
    
    return score
