# CLAUDE.md — project context & Mac (Apple Silicon) setup

This file is auto-loaded by Claude Code. If you are a Claude agent picking this repo up on
a new machine (the owner is moving from a Windows RTX 3050 box to an **M4 Pro Mac**), read
this first, then run the setup + retrain steps below. Everything runs **fully offline** after
dependencies are installed — the project forbids external LLM API calls anywhere.

## What this project is

An empirical study: **regex/heuristic prompt-injection detection vs. a fine-tuned DistilBERT
classifier**, both scoring MCP (Model Context Protocol) tool outputs. Research question: does
the learned classifier generalize to *rephrased / novel* injections better than the heuristic
approach real MCP guards use (which only catch known patterns)?

**Core result already demonstrated** (on Windows CPU): on held-out paraphrased attacks the
regex baseline scores **0.19 accuracy (0% recall)** while DistilBERT scores **0.95 accuracy /
0.94 recall / 0% FPR**. Both are perfect on seen patterns. The gap is the thesis.

## Layout

```
src/mcp_guard/
  dataset/        generate_benign.py, generate_injections.py, generate_held_out.py, common.py
  detectors/      baseline_detector.py (regex, 6 categories), ml_detector.py (DistilBERT)
  training/       train.py, config.py        device-adaptive (CUDA / MPS / CPU)
  evaluation/     evaluate.py                 writes results/metrics.csv + chart
  proxy/          proxy_server.py (FastAPI), db.py (SQLite WAL), demo_client.py
  dashboard/      app.py (Streamlit)
  device.py       best_device(): cuda -> mps -> cpu
tests/            test_baseline.py, test_proxy.py
data/             seen/ (258 inj), benign/ (290), held_out/ (255 paraphrases)  [committed]
models/           distilbert-finetuned/   [gitignored — RETRAIN on the Mac]
results/          metrics.csv, *.png       [regenerable]
```

`data/` is committed, so you do **not** need to regenerate it. `models/` is gitignored
(268MB, over GitHub's limit), so the model **must be retrained** on the Mac — see below.

## Mac (M4 Pro) setup — first time

The repo uses `uv`. Torch source is platform-aware (see pyproject.toml): on macOS it resolves
the default PyPI wheel, which includes **Metal/MPS GPU support**, so training auto-accelerates.

```bash
# 1. install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: brew install uv

# 2. install deps (creates .venv, pulls MPS-capable torch)
uv sync --extra train --extra eval --extra serve

# 3. sanity: confirm MPS is picked up
uv run python -c "import torch; print('mps available:', torch.backends.mps.is_available())"
```

## Retrain + verify on the Mac

```bash
# Retrain (device.py auto-selects MPS; ~1-3 min on M4 Pro vs ~11 min Windows CPU)
uv run python -m mcp_guard.training.train

# Regenerate the comparison table + chart
uv run python -m mcp_guard.evaluation.evaluate      # -> results/metrics.csv, accuracy_gap_chart.png

# Tests (baseline + proxy, loads the model)
uv run pytest -q

# Live divergence demo
uv run uvicorn mcp_guard.proxy.proxy_server:app --port 8000   # terminal 1
uv run python -m mcp_guard.proxy.demo_client                  # terminal 2

# Dashboard
uv run streamlit run src/mcp_guard/dashboard/app.py
```

Expected after retrain: seen split ~1.0 F1 for both detectors; held-out split regex ~0.19
accuracy vs DistilBERT ~0.95. Exact DistilBERT numbers vary slightly by run/seed — that's fine.

## Conventions / gotchas

- **No external LLM API calls** anywhere (dataset paraphrases are hand-authored literals in
  `generate_held_out.py`; detection is fully local). Keep it that way.
- Regenerating data is only needed if you change the generators:
  `uv run python -m mcp_guard.dataset.generate_benign` (and `...generate_injections`,
  `...generate_held_out`). Held-out paraphrases are deliberately worded to evade the regex.
- Offline runs: set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to stop HuggingFace pinging the
  hub once the base model is cached.
- The 16 paraphrases DistilBERT misses are the novel-encoding ones (hex/base32/rot13/
  homoglyph) — a genuine, documented limitation, not a bug.

## Deferred / enrichment ideas (not yet done)

McNemar significance test; threshold calibration curves; more encoding-attack coverage; a
real MCP-client integration for the proxy (current demo posts tool outputs over HTTP).
