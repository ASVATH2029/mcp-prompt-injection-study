"""Device selection shared by training and inference.

Prefers CUDA (NVIDIA), then MPS (Apple Silicon Metal), then CPU. This lets the same code
accelerate on an RTX 3050 (Windows/Linux + CUDA wheel) and on an M-series Mac (MPS), and
fall back to CPU anywhere else.
"""

from __future__ import annotations

import torch


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple MPS (Metal)"
    return "CPU"
