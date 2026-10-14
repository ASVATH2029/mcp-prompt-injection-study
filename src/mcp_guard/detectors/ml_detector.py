"""Inference wrapper around the fine-tuned DistilBERT classifier.

Loads a saved checkpoint and exposes the same predict() interface as BaselineDetector,
so evaluation and the proxy can treat both detectors uniformly. Runs on GPU if present,
else CPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mcp_guard.detectors.baseline_detector import DetectorResult
from mcp_guard.device import best_device
from mcp_guard.training.config import MODEL_DIR


class MLDetector:
    def __init__(self, model_dir: str | Path = MODEL_DIR, max_length: int = 256) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No model at {model_dir}. Run `uv run python -m mcp_guard.training.train` first."
            )
        self.device = best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.inference_mode()
    def predict(self, text: str) -> DetectorResult:
        if not text:
            return DetectorResult(label=0, confidence=0.0)
        inputs = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        label = int(torch.argmax(probs).item())
        confidence = float(probs[label].item())
        return DetectorResult(label=label, confidence=confidence, matched_rules=["distilbert"])
