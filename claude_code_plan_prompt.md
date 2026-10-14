# Claude Code Plan Prompt — MCP Prompt-Injection Generalization Study

Copy everything below into Claude Code (in plan mode) to kick off the project.

---

## Prompt

I'm building a final-year CS project: an empirical study comparing a regex/heuristic
prompt-injection detector against a fine-tuned ML classifier, both operating on MCP
(Model Context Protocol) tool outputs. The core research question: does a learned
classifier generalize to rephrased/novel injection patterns better than the heuristic
approach every existing MCP guard uses (which explicitly admits it only catches known
patterns, not paraphrased ones).

Build this in stages. Enter plan mode first and propose the file/module structure
before writing code.

### Scope for this session (2-day MVP)

1. **Dataset construction**

   - A script that pulls/simulates benign tool outputs from 2-3 real open-source MCP
     servers (filesystem server, a simple web-fetch server — use the official MCP
     Python SDK examples as a base).
   - A script that generates labeled injection examples in two tiers:
     - `seen/`: known injection templates (ignore-instruction phrases, fake system
       messages, base64-encoded payloads) — this is the training set.
     - `held_out/`: semantically rephrased variants of the same attacks — this is the
       evaluation-only set, never touched during training. Generate these yourself
       (as Claude Code) directly in the conversation: for each seen-tier template,
       write ~10 paraphrased variants with genuinely different wording/structure
       (synonyms, reordering, indirect phrasing, different encodings) and save them
       straight to the held_out/ file. Do NOT call any external API for this — no
       Claude API, no other LLM API. This is a one-time manual-authorship step done
       by you as the coding assistant, not a scripted API call.
   - Output as labeled JSONL: `{text, label, source, tier}`.
2. **Baseline detector**

   - A regex/heuristic module (`baseline_detector.py`) replicating the pattern
     categories used by real MCP guards (e.g. Agent Guard, bulwark-mcp): instruction-
     override phrases, base64 payload detection, suspicious encoding/markdown tricks.
   - This must be a clean, inspectable rule list — not a black box — since it's the
     comparison arm.
3. **Learned detector**

   - Fine-tune DistilBERT (HuggingFace `transformers`) for binary classification
     (injected vs benign) on the `seen` tier only.
   - Keep it runnable on a single consumer GPU (RTX 3050 6GB) — use small batch size /
     gradient accumulation if needed, document the training config.
4. **Evaluation**

   - Run both detectors on: seen-tier held-out split, and the held_out/ rephrased tier.
   - Report accuracy, precision, recall, F1, and false-positive rate for both, on both
     splits. This comparison table is the core result — make it easy to regenerate.
   - Save results to a results/ directory as CSV + a simple bar chart (matplotlib is
     fine) showing the seen-vs-held-out accuracy gap for each detector side by side.
5. **Runtime proxy (the demo)**

   - A lightweight MCP proxy (stdio or FastAPI HTTP, your call — propose which fits
     better) that sits between an MCP client and a real MCP server, scores every tool
     output with BOTH detectors, logs the verdicts (SQLite, JSONL rows: tool, output,
     regex_verdict, model_verdict, confidence, latency), and optionally blocks above
     a threshold.
   - Must be demoable live: feed it a known-pattern injection and a rephrased one,
     show the regex/model verdicts diverge on the rephrased case.
6. **Demo dashboard**

   - Streamlit app reading from the SQLite log: a live/recent table of tool calls with
     both verdicts side by side, highlighting disagreements, plus the aggregate
     accuracy/F1 chart from step 4.

### Constraints

- Python throughout. Use `uv` or `venv` for environment management — propose which.
- No external LLM API calls anywhere in this project — not in dataset generation, not
  in the live detection path. The held-out paraphrases are hand-authored by you
  (Claude Code) directly as part of the build, and the deployed proxy runs a fully
  local/offline classifier at inference time. This keeps the whole pipeline free of
  any paid dependency and keeps the generalization claim clean (a fast, self-contained
  local detector, not another LLM call).
- Keep the regex baseline genuinely representative of real tools' approaches, not a
  strawman — if it's too weak, the comparison isn't credible.
- Every stage should produce something runnable and testable before moving to the
  next — I want to demo progress incrementally, not get a wall of code at the end.

### What I want from you right now

Propose the project structure (directories, key files, dependencies) and a step order
for building this across the stages above. Flag anything in my plan that's unrealistic
for a 2 to 1-day build, and tell me what you'd defer to a later "enrichment" phase instead.
Don't start writing implementation code until I approve the plan.
