"""Shared helpers for dataset scripts: JSONL IO and record schema."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Repo root = three levels up from this file (src/mcp_guard/dataset/common.py).
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"


def make_record(text: str, label: int, source: str, tier: str) -> dict[str, Any]:
    """Build one labeled example. label: 1=injection, 0=benign."""
    return {"text": text, "label": label, "source": source, "tier": tier}


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
