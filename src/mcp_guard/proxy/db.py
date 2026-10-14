"""SQLite logging for proxy verdicts (stdlib sqlite3, WAL mode).

WAL lets the Streamlit dashboard read the log concurrently while the proxy writes.
One row per scored tool output.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from mcp_guard.dataset.common import REPO_ROOT

_DEFAULT_DB_PATH = REPO_ROOT / "logs" / "proxy.db"


def db_path() -> Path:
    """Resolve the DB path, honoring the MCP_GUARD_DB override (used by tests)."""
    env = os.environ.get("MCP_GUARD_DB")
    return Path(env) if env else _DEFAULT_DB_PATH


# Back-compat module attribute for callers/tests that read db.DB_PATH directly.
DB_PATH = db_path()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proxy_logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL,
    tool           TEXT    NOT NULL,
    output_snippet TEXT    NOT NULL,
    regex_verdict  INTEGER NOT NULL,
    regex_conf     REAL    NOT NULL,
    model_verdict  INTEGER,
    model_conf     REAL,
    disagree       INTEGER NOT NULL,
    blocked        INTEGER NOT NULL,
    latency_ms     REAL    NOT NULL
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(_SCHEMA)


def insert_log(
    conn: sqlite3.Connection,
    *,
    tool: str,
    output_snippet: str,
    regex_verdict: int,
    regex_conf: float,
    model_verdict: int | None,
    model_conf: float | None,
    blocked: bool,
    latency_ms: float,
) -> None:
    disagree = int(model_verdict is not None and regex_verdict != model_verdict)
    conn.execute(
        """
        INSERT INTO proxy_logs
          (ts, tool, output_snippet, regex_verdict, regex_conf,
           model_verdict, model_conf, disagree, blocked, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(timespec="seconds"),
            tool,
            output_snippet[:500],
            regex_verdict,
            regex_conf,
            model_verdict,
            model_conf,
            disagree,
            int(blocked),
            latency_ms,
        ),
    )
    conn.commit()


def recent(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM proxy_logs ORDER BY id DESC LIMIT ?", (limit,))
    return cur.fetchall()
