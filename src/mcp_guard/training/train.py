"""Fine-tune DistilBERT on the SEEN tier only (train split).

Trains a binary injected/benign classifier and saves it to models/distilbert-finetuned/.
The held-out paraphrase tier is never seen here — it exists only for evaluation.

Device-adaptive: uses GPU with fp16 + gradient accumulation when CUDA is available,
otherwise runs on CPU with a plain config. Works today on CPU; switch the torch index
to cu124 (see pyproject.toml) and re-run to use the RTX 3050.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from mcp_guard.data_splits import Example, build_splits
from mcp_guard.device import best_device, device_label
from mcp_guard.training.config import TrainConfig


def _to_dataset(examples: list[Example]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [e.text for e in examples],
            "label": [e.label for e in examples],
        }
    )


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main() -> None:
    cfg = TrainConfig()
    device = best_device()
    print(f"Training device: {device_label(device)}")

    splits = build_splits(cfg.seed)
    print(
        f"train={len(splits.train)}  eval_seen={len(splits.eval_seen)}  "
        f"(held-out is eval-only, not used in training)"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    train_ds = _to_dataset(splits.train).map(tokenize, batched=True)
    eval_ds = _to_dataset(splits.eval_seen).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=2, id2label={0: "benign", 1: "injected"},
        label2id={"benign": 0, "injected": 1},
    )

    if device == "cuda":
        # RTX 3050 6GB: small batch + accumulation + fp16.
        batch_size, grad_accum, fp16 = cfg.gpu_batch_size, cfg.gpu_grad_accum, True
    elif device == "mps":
        # Apple Silicon: larger batch fits in unified memory; fp16 is unstable on MPS.
        batch_size, grad_accum, fp16 = cfg.mps_batch_size, 1, False
    else:
        batch_size, grad_accum, fp16 = cfg.cpu_batch_size, 1, False

    # transformers 5.x dropped warmup_ratio; derive warmup_steps from total steps.
    steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * cfg.num_train_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        seed=cfg.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval (seen split):", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    print(f"Saved model -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
