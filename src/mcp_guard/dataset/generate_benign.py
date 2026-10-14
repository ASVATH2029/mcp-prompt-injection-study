"""Generate BENIGN tool-output examples (label 0).

Two MCP server archetypes are simulated:
  * filesystem server  -> read_file / list_directory / file_info results, drawn from
    REAL text files (Python stdlib source + this repo) so the content is authentic
    and diverse rather than hand-waved.
  * web-fetch server    -> fetched HTML / JSON / markdown / article bodies, synthesized
    from realistic templates (kept offline — no network required).

Output: data/benign/benign.jsonl. Target ~250 records to roughly balance the injection
class. Fully offline, no API calls.
"""

from __future__ import annotations

import argparse
import os
import random
import sysconfig
from pathlib import Path

from mcp_guard.dataset.common import DATA_DIR, make_record, write_jsonl

REPO_ROOT = DATA_DIR.parent
MAX_CHARS = 1200  # cap each sample so one giant file doesn't dominate


def _gather_text_files() -> list[Path]:
    """Collect real text files from the stdlib and this repo."""
    roots: list[Path] = []
    stdlib = Path(sysconfig.get_path("stdlib"))
    if stdlib.exists():
        roots.append(stdlib)
    roots.append(REPO_ROOT / "src")

    files: list[Path] = []
    exts = {".py", ".txt", ".cfg", ".toml", ".md", ".json", ".ini"}
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in exts and "site-packages" not in p.parts:
                files.append(p)
            if len(files) > 4000:
                break
    return files


def _read_snippet(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return None
    text = text.strip()
    if not text:
        return None
    return text[:MAX_CHARS]


def filesystem_records(rng: random.Random, n: int) -> list[dict]:
    files = _gather_text_files()
    rng.shuffle(files)
    records: list[dict] = []
    for path in files:
        if len(records) >= n:
            break
        snippet = _read_snippet(path)
        if snippet is None:
            continue
        kind = rng.choice(["read_file", "read_file", "file_info"])
        if kind == "read_file":
            text = f"Tool result read_file({path.name!r}):\n{snippet}"
        else:
            try:
                size = path.stat().st_size
            except OSError:
                size = len(snippet)
            text = (
                f"Tool result file_info({path.name!r}): "
                f"size={size} bytes, type=text, lines~{snippet.count(chr(10)) + 1}"
            )
        records.append(make_record(text, label=0, source="benign:filesystem", tier="seen"))

    # Directory listings from real dirs.
    dirs = {p.parent for p in files}
    for d in list(dirs):
        if len(records) >= n + 20:
            break
        try:
            entries = sorted(os.listdir(d))[:12]
        except OSError:
            continue
        if not entries:
            continue
        listing = ", ".join(entries)
        text = f"Tool result list_directory({d.name!r}): {listing}"
        records.append(make_record(text, label=0, source="benign:filesystem", tier="seen"))
    return records


# --- Web-fetch archetypes (synthetic but realistic) -------------------------------

_HTML = [
    (
        "<html><head><title>{title}</title></head><body><h1>{title}</h1>"
        "<p>{para}</p><p>{para2}</p></body></html>"
    ),
    "<article><h2>{title}</h2><p>{para}</p><ul><li>{para2}</li></ul></article>",
]
_JSON = [
    '{{"status": "ok", "data": {{"id": {id}, "name": "{title}", "active": true}}}}',
    '{{"results": [{{"title": "{title}", "score": {id}}}], "page": 1, "total": {id}}}',
    '{{"temperature": {id}, "unit": "celsius", "city": "{title}", "humidity": 60}}',
]
_MD = [
    "# {title}\n\n{para}\n\n## Details\n\n- {para2}\n- Updated recently",
    "## {title}\n\n{para}\n\nSee the docs for more on {title}.",
]
_TITLES = [
    "Quarterly Report", "Getting Started Guide", "Release Notes", "Weather Forecast",
    "Product Catalog", "API Reference", "User Profile", "Search Results",
    "Installation Steps", "Changelog", "Dashboard Metrics", "FAQ",
]
_PARAS = [
    "This document summarizes the key findings from the latest analysis.",
    "The service returned a healthy status with nominal latency across all regions.",
    "Follow the steps below to configure your environment correctly.",
    "Our team has released several improvements to performance and stability.",
    "The dataset contains anonymized records collected over the past quarter.",
    "Review the attached figures for a breakdown of usage by category.",
    "All endpoints are operational and responding within expected thresholds.",
    "The guide walks through setup, configuration, and basic usage patterns.",
]


def webfetch_records(rng: random.Random, n: int) -> list[dict]:
    records: list[dict] = []
    for _ in range(n):
        template = rng.choice(_HTML + _JSON + _MD)
        body = template.format(
            title=rng.choice(_TITLES),
            para=rng.choice(_PARAS),
            para2=rng.choice(_PARAS),
            id=rng.randint(1, 500),
        )
        url = f"https://example.{rng.choice(['com', 'org', 'io'])}/{rng.choice(['docs', 'api', 'blog', 'data'])}"
        text = f"Fetched {url} — body:\n{body}"
        records.append(make_record(text, label=0, source="benign:webfetch", tier="seen"))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fs", type=int, default=150, help="filesystem records")
    parser.add_argument("--n-web", type=int, default=120, help="web-fetch records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = filesystem_records(rng, args.n_fs) + webfetch_records(rng, args.n_web)
    rng.shuffle(records)

    out = DATA_DIR / "benign" / "benign.jsonl"
    n = write_jsonl(out, records)
    by_src: dict[str, int] = {}
    for r in records:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"Wrote {n} benign records -> {out}")
    for src, count in sorted(by_src.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
