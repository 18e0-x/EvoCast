# -*- coding: utf-8 -*-
"""
Device utility functions for PyTorch operations.
This module has no dependencies to avoid circular imports.
"""
import os

import torch


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Checks for device availability in the following order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs)
    3. CPU (fallback)
    
    :return: torch.device object representing the best available device
    """
    if str(os.environ.get("TFB_FORCE_CPU") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
