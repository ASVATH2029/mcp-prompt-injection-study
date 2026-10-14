"""FastAPI MCP proxy: scores every tool output with BOTH detectors.

Sits between an MCP client and server. For each tool output it:
  1. runs the regex baseline and (if trained) the DistilBERT classifier,
  2. logs both verdicts + latency to SQLite,
  3. optionally blocks (HTTP 403) when a detector flags above the block threshold.

The divergence endpoint is the live demo: a known-pattern injection trips both
detectors; a rephrased one trips only the classifier.

Run:  uv run uvicorn mcp_guard.proxy.proxy_server:app --port 8000
Env:  MCP_GUARD_BLOCK_THRESHOLD (default 0.85), MCP_GUARD_BLOCK (default "1")
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mcp_guard.detectors import BaselineDetector
from mcp_guard.proxy import db

BLOCK_THRESHOLD = float(os.environ.get("MCP_GUARD_BLOCK_THRESHOLD", "0.85"))
BLOCK_ENABLED = os.environ.get("MCP_GUARD_BLOCK", "1") == "1"

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    _state["regex"] = BaselineDetector()
    _state["ml"] = None
    try:
        from mcp_guard.detectors.ml_detector import MLDetector

        _state["ml"] = MLDetector()
        print("Loaded DistilBERT detector.")
    except FileNotFoundError:
        print("No trained model yet — proxy runs with regex baseline only.")
    yield
    _state.clear()


app = FastAPI(title="MCP Guard Proxy", lifespan=lifespan)


class ToolOutput(BaseModel):
    tool_name: str
    output_text: str


class Verdict(BaseModel):
    tool_name: str
    regex_verdict: int
    regex_confidence: float
    regex_rules: list[str]
    model_verdict: int | None
    model_confidence: float | None
    disagreement: bool
    blocked: bool
    latency_ms: float


def _score(payload: ToolOutput) -> Verdict:
    regex: BaselineDetector = _state["regex"]  # type: ignore[assignment]
    ml = _state["ml"]

    start = time.perf_counter()
    r = regex.predict(payload.output_text)
    model_verdict: int | None = None
    model_conf: float | None = None
    if ml is not None:
        m = ml.predict(payload.output_text)
        model_verdict, model_conf = m.label, m.confidence
    latency_ms = (time.perf_counter() - start) * 1000

    disagreement = model_verdict is not None and r.label != model_verdict
    flagged_high = r.label == 1 or (
        model_verdict == 1 and (model_conf or 0.0) >= BLOCK_THRESHOLD
    )
    blocked = BLOCK_ENABLED and flagged_high

    with db.connect() as conn:
        db.insert_log(
            conn,
            tool=payload.tool_name,
            output_snippet=payload.output_text,
            regex_verdict=r.label,
            regex_conf=r.confidence,
            model_verdict=model_verdict,
            model_conf=model_conf,
            blocked=blocked,
            latency_ms=latency_ms,
        )

    return Verdict(
        tool_name=payload.tool_name,
        regex_verdict=r.label,
        regex_confidence=round(r.confidence, 4),
        regex_rules=r.matched_rules,
        model_verdict=model_verdict,
        model_confidence=round(model_conf, 4) if model_conf is not None else None,
        disagreement=disagreement,
        blocked=blocked,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state.get("ml") is not None}


@app.post("/tool-output")
def tool_output(payload: ToolOutput) -> JSONResponse:
    verdict = _score(payload)
    status = 403 if verdict.blocked else 200
    return JSONResponse(status_code=status, content=verdict.model_dump())
