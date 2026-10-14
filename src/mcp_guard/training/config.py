"""Training hyperparameters. Documented for reproducibility (see plan Stage 3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcp_guard.dataset.common import REPO_ROOT

MODEL_DIR = REPO_ROOT / "models" / "distilbert-finetuned"


@dataclass
class TrainConfig:
    base_model: str = "distilbert-base-uncased"
    max_length: int = 256
    num_train_epochs: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42

    # train.py adapts batch/precision to the actual device.
    # CUDA (RTX 3050 6GB): batch 8 x grad-accum 4 = effective 32, fp16 on.
    # MPS (Apple Silicon, e.g. M4 Pro): batch 16 in unified memory, no fp16.
    # CPU: batch 8, grad-accum 1, no fp16 (fp16 gives no CPU benefit).
    gpu_batch_size: int = 8
    gpu_grad_accum: int = 4
    mps_batch_size: int = 16
    cpu_batch_size: int = 8

    output_dir: Path = MODEL_DIR
