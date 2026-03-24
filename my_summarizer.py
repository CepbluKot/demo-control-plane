from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple


def _normalize_period(
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> Tuple[str, str]:
    """
    Supports both call styles used by control_plane.processing:
    - period_start / period_end (ISO strings)
    - start_dt / end_dt (datetime)
    """
    if period_start and period_end:
        return period_start, period_end
    if start_dt is not None and end_dt is not None:
        return start_dt.isoformat(), end_dt.isoformat()
    raise ValueError("Provide either period_start+period_end or start_dt+end_dt")


def summarize_logs(
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    anomaly: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Template summarizer adapter.
    Replace this function body with your real PeriodLogSummarizer integration.

    Expected return format:
    - "summary": final text
    - one of map arrays for UI live batches:
      map_summaries | batch_summaries | chunk_summaries | partial_summaries
    """
    start_iso, end_iso = _normalize_period(
        period_start=period_start,
        period_end=period_end,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    anomaly_ts = (anomaly or {}).get("timestamp", "unknown")

    # TODO: replace with your real implementation.
    # Example integration point:
    # 1) build db_fetch_page(columns, period_start, period_end, limit, offset)
    # 2) build llm_call(prompt)
    # 3) run PeriodLogSummarizer(...).summarize_period(...)
    # 4) return {"summary": result.summary, "chunk_summaries": result.chunk_summaries}
    map_batches = [
        f"[MAP 1] Loaded logs around anomaly {anomaly_ts}.",
        "[MAP 2] Error-like signals were found in several log rows.",
        "[MAP 3] Most likely issue cluster identified (template output).",
    ]
    final_summary = (
        f"[TEMPLATE] Summary for period [{start_iso}, {end_iso}) "
        f"around anomaly {anomaly_ts}. "
        "Replace my_summarizer.summarize_logs with your production summarizer."
    )

    return {
        "summary": final_summary,
        "chunk_summaries": map_batches,
        "source": "my_summarizer_template",
    }
