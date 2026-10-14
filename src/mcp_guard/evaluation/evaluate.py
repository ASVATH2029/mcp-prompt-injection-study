"""Run both detectors on both eval splits and write the comparison table + chart.

Core result of the study: for each (split, detector) pair, report accuracy, precision,
recall, F1, and false-positive rate. The seen-vs-held-out accuracy gap is the headline
figure — the regex baseline is expected to drop sharply on held-out paraphrases while
the classifier should hold up better.

Outputs:
  results/metrics.csv
  results/accuracy_gap_chart.png
"""

from __future__ import annotations

import csv

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from mcp_guard.data_splits import Example, build_splits
from mcp_guard.dataset.common import REPO_ROOT
from mcp_guard.detectors import BaselineDetector

RESULTS_DIR = REPO_ROOT / "results"


def _metrics(detector, examples: list[Example]) -> dict[str, float]:
    y_true = [e.label for e in examples]
    y_pred = [detector.predict(e.text).label for e in examples]
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
    }


def _load_detectors() -> dict[str, object]:
    detectors: dict[str, object] = {"regex_baseline": BaselineDetector()}
    try:
        from mcp_guard.detectors.ml_detector import MLDetector

        detectors["distilbert"] = MLDetector()
    except FileNotFoundError:
        print("WARNING: no trained model found — evaluating baseline only.")
    return detectors


def run() -> list[dict]:
    splits = build_splits()
    eval_sets = {"seen": splits.eval_seen, "held_out": splits.eval_held_out}
    detectors = _load_detectors()

    rows: list[dict] = []
    for split_name, examples in eval_sets.items():
        for det_name, det in detectors.items():
            m = _metrics(det, examples)
            rows.append({"split": split_name, "detector": det_name, **m})
    return rows


def write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "metrics.csv"
    fields = ["split", "detector", "accuracy", "precision", "recall", "f1", "fpr"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {out}")


def write_chart(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = sorted({r["detector"] for r in rows})
    splits = ["seen", "held_out"]
    acc = {(r["split"], r["detector"]): r["accuracy"] for r in rows}

    x = range(len(detectors))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"seen": "#4C78A8", "held_out": "#E45756"}
    for i, split in enumerate(splits):
        heights = [acc.get((split, d), 0.0) for d in detectors]
        offs = [xi + (i - 0.5) * width for xi in x]
        bars = ax.bar(offs, heights, width, label=split, color=colors[split])
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(detectors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Seen vs. held-out accuracy by detector")
    ax.legend(title="eval split")
    fig.tight_layout()
    out = RESULTS_DIR / "accuracy_gap_chart.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def main() -> None:
    rows = run()
    print(f"\n{'split':<10}{'detector':<18}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'fpr':>7}")
    for r in rows:
        print(
            f"{r['split']:<10}{r['detector']:<18}"
            f"{r['accuracy']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}"
            f"{r['f1']:>7.3f}{r['fpr']:>7.3f}"
        )
    print()
    write_csv(rows)
    write_chart(rows)


if __name__ == "__main__":
    main()
