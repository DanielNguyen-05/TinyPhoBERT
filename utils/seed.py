"""
utils/seed.py
Reproducibility utilities.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For deterministic behavior on CUDA (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device
