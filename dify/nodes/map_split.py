"""Dify Code Node: MAP Split

Splits log rows into N equal chunks for parallel MAP execution.
The output array feeds into an Iteration node (parallel=true),
each item becomes the user message for one LLM MAP call.

IN:
  logs           (str) — JSON string: list of log row objects from ClickHouse.
                         Each row must have "raw_line" and optionally "timestamp".
  n              (int) — number of chunks (= number of parallel LLM calls).
  incident_start (str) — ISO8601 incident start, e.g. "2026-03-18T01:00:00"
  incident_end   (str) — ISO8601 incident end

OUT:
  batches        (array[str]) — N formatted text blocks, one per LLM call.
  chunk_count    (int)        — actual chunks produced (≤ n).

Wire-up:
  Iteration node → input array: map_split.batches
  LLM node inside Iteration → user message: {{#item#}}
"""

import json
import math


def main(logs: str, n: int, incident_start: str = "", incident_end: str = "") -> dict:
    try:
        rows = json.loads(logs) if isinstance(logs, str) else list(logs)
    except (json.JSONDecodeError, ValueError):
        return {"batches": [], "chunk_count": 0}

    if not rows or n <= 0:
        return {"batches": [], "chunk_count": 0}

    n = min(n, len(rows))
    size = math.ceil(len(rows) / n)

    chunks = [rows[i : i + size] for i in range(0, len(rows), size)]
    batches = [_format_chunk(chunk, incident_start, incident_end) for chunk in chunks]

    return {"batches": batches, "chunk_count": len(batches)}


# ── helpers ──────────────────────────────────────────────────────────────────

def _zone(ts: str, start: str, end: str) -> str:
    """Return zone prefix based on ISO8601 string comparison."""
    if not ts or not start or not end:
        return "INC"
    t = ts[:19]
    if t < start[:19]:
        return "CB"
    if t > end[:19]:
        return "CA"
    return "INC"


def _format_chunk(rows: list, incident_start: str, incident_end: str) -> str:
    """Format a list of log row dicts as a single text block for the LLM."""
    lines = []
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
            continue
        raw = row.get("raw_line") or json.dumps(row, ensure_ascii=False)
        ts = str(
            row.get("timestamp") or row.get("time") or row.get("ts") or ""
        )
        prefix = _zone(ts, incident_start, incident_end)
        lines.append(f"[{prefix}] {raw}")
    return "\n".join(lines)
