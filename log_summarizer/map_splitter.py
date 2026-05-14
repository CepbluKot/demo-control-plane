"""map_splitter — split log rows into N chunks for parallel MAP execution.

Replaces token-budget chunking when you want a fixed number of parallel
LLM calls (e.g. N servers, N workers).

Example:
    from log_summarizer.map_splitter import split_for_map
    from log_summarizer.map_processor import MapProcessor

    chunks = split_for_map(all_rows, n=4)

    results = await asyncio.gather(*[
        map_processor.process_chunk(chunk) for chunk in chunks
    ])
    # results: list[list[BatchAnalysis]], flatten before REDUCE
"""

from __future__ import annotations

import math
from typing import Sequence

from log_summarizer.models import Chunk, LogRow
from log_summarizer.utils.tokens import estimate_tokens


def split_for_map(rows: Sequence[LogRow], n: int) -> list[Chunk]:
    """Split rows into exactly N equal-size chunks.

    Splits by row count, not token count — each LLM call gets the same
    number of log lines regardless of their verbosity.

    Args:
        rows: All log rows in chronological order.
        n:    Number of chunks (= number of parallel LLM calls).
              Capped at len(rows) so you never get empty chunks.

    Returns:
        List of at most N Chunk objects ready for MapProcessor.process_chunk().
    """
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")
    if not rows:
        return []

    n = min(n, len(rows))
    size = math.ceil(len(rows) / n)

    return [_make_chunk(list(rows[i : i + size])) for i in range(0, len(rows), size)]


def _make_chunk(rows: list[LogRow]) -> Chunk:
    zones = {r.zone for r in rows}
    batch_zone = zones.pop() if len(zones) == 1 else "mixed"

    first_ts = rows[0].timestamp
    last_ts = rows[-1].timestamp
    if first_ts > last_ts:
        first_ts, last_ts = last_ts, first_ts

    token_estimate = sum(estimate_tokens(r.raw_line) for r in rows)

    return Chunk(
        rows=rows,
        time_range=(first_ts, last_ts),
        token_estimate=token_estimate,
        batch_zone=batch_zone,
    )
