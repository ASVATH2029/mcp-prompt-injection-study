"""Deterministic dataset splits shared by training and evaluation.

Layout:
  * seen injections + benign  -> stratified 80/20.
      - 80% (inj_train + benign_train) is the TRAINING set.
      - 20% (inj_eval_seen + benign_eval) is the SEEN-split eval set.
  * held-out paraphrases + benign_eval -> the HELD-OUT eval set.

benign_eval is reused across both eval sets and is never trained on, so false-positive
rates are measured on held-out benign in both cases. A fixed seed makes every split
reproducible, which keeps the comparison table regenerable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mcp_guard.dataset.common import DATA_DIR, read_jsonl

SEED = 42
TRAIN_FRACTION = 0.8


@dataclass
class Example:
    text: str
    label: int
    source: str


def _load(path_parts: tuple[str, ...]) -> list[Example]:
    path = DATA_DIR.joinpath(*path_parts)
    return [
        Example(text=r["text"], label=int(r["label"]), source=r["source"])
        for r in read_jsonl(path)
    ]


def _split(items: list[Example], seed: int) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * TRAIN_FRACTION)
    return shuffled[:cut], shuffled[cut:]


@dataclass
class Splits:
    train: list[Example]
    eval_seen: list[Example]
    eval_held_out: list[Example]


def build_splits(seed: int = SEED) -> Splits:
    injections = _load(("seen", "injections.jsonl"))
    benign = _load(("benign", "benign.jsonl"))
    held_out = _load(("held_out", "paraphrases.jsonl"))

    inj_train, inj_eval = _split(injections, seed)
    ben_train, ben_eval = _split(benign, seed + 1)

    train = inj_train + ben_train
    eval_seen = inj_eval + ben_eval
    eval_held_out = held_out + ben_eval

    rng = random.Random(seed + 2)
    rng.shuffle(train)
    return Splits(train=train, eval_seen=eval_seen, eval_held_out=eval_held_out)


def summary() -> str:
    s = build_splits()

    def counts(items: list[Example]) -> str:
        pos = sum(e.label for e in items)
        return f"{len(items)} total ({pos} injected / {len(items) - pos} benign)"

    return (
        f"train:         {counts(s.train)}\n"
        f"eval_seen:     {counts(s.eval_seen)}\n"
        f"eval_held_out: {counts(s.eval_held_out)}"
    )


if __name__ == "__main__":
    print(summary())
