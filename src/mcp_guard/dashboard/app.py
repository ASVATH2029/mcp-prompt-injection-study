"""MCP Guard dashboard.

Tab 1 (Live Log): recent proxy verdicts from SQLite, with disagreements (regex vs.
model) highlighted — the visual proof that the two detectors diverge on paraphrases.
Tab 2 (Accuracy Report): the seen-vs-held-out comparison table + chart from results/.

Run:  uv run streamlit run src/mcp_guard/dashboard/app.py
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from mcp_guard.dataset.common import REPO_ROOT
from mcp_guard.proxy.db import db_path

DB_PATH = db_path()
RESULTS_DIR = REPO_ROOT / "results"

st.set_page_config(page_title="MCP Injection Monitor", layout="wide")
st.title("MCP Injection Monitor")
st.caption("Regex baseline vs. fine-tuned DistilBERT on MCP tool outputs")

tab_log, tab_report = st.tabs(["Live Log", "Accuracy Report"])


def _load_logs() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    # read-only connection so we never contend with the proxy's writes
    uri = f"file:{DB_PATH}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(
            "SELECT id, ts, tool, regex_verdict, model_verdict, model_conf, "
            "disagree, blocked, latency_ms, output_snippet "
            "FROM proxy_logs ORDER BY id DESC LIMIT 200",
            conn,
        )
    except pd.errors.DatabaseError:
        return pd.DataFrame()
    finally:
        conn.close()


with tab_log:
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    df = _load_logs()
    if df.empty:
        st.info("No proxy activity yet. Start the proxy and send a tool output to it.")
    else:
        verdict_label = {0: "benign", 1: "INJECTED"}
        show = df.copy()
        show["regex"] = show["regex_verdict"].map(verdict_label)
        show["model"] = show["model_verdict"].map(
            lambda v: "—" if pd.isna(v) else verdict_label[int(v)]
        )
        show = show[
            ["id", "ts", "tool", "regex", "model", "model_conf",
             "disagree", "blocked", "latency_ms", "output_snippet"]
        ]

        def _highlight(row: pd.Series):
            if row["disagree"]:
                return ["background-color: #FFF3CD"] * len(row)  # amber: detectors differ
            if row["blocked"]:
                return ["background-color: #F8D7DA"] * len(row)  # red: blocked
            return [""] * len(row)

        n_dis = int(df["disagree"].sum())
        n_block = int(df["blocked"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Logged calls", len(df))
        c2.metric("Disagreements", n_dis)
        c3.metric("Blocked", n_block)
        st.dataframe(
            show.style.apply(_highlight, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Amber = regex and model disagree (the generalization gap). Red = blocked.")


with tab_report:
    csv_path = RESULTS_DIR / "metrics.csv"
    chart_path = RESULTS_DIR / "accuracy_gap_chart.png"
    if not csv_path.exists():
        st.info("No results yet. Run `uv run python -m mcp_guard.evaluation.evaluate`.")
    else:
        metrics = pd.read_csv(csv_path)
        st.subheader("Metrics by split and detector")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        if chart_path.exists():
            st.subheader("Seen vs. held-out accuracy")
            st.image(str(chart_path))
        st.caption(
            "The regex baseline's accuracy collapses on the held-out (paraphrased) split "
            "while the classifier is expected to generalize further — the study's core result."
        )
