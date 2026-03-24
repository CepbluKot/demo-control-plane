import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


def _safe_preview(value: Any, max_len: int = 200) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        rendered = str(value)
    if len(rendered) > max_len:
        return f"{rendered[:max_len]}..."
    return rendered


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.DEBUG,
    **fields: Any,
) -> None:
    if not logger.isEnabledFor(level):
        return
    payload = ", ".join(f"{k}={_safe_preview(v)}" for k, v in fields.items())
    logger.log(level, "%s | %s", event, payload)


def log_dataframe(
    logger: logging.Logger,
    name: str,
    df: pd.DataFrame,
    level: int = logging.DEBUG,
    preview_rows: int = 3,
) -> None:
    if not logger.isEnabledFor(level):
        return
    if df is None:
        logger.log(level, "DataFrame[%s] is None", name)
        return
    shape = df.shape
    columns = list(df.columns)
    extra: Dict[str, Any] = {"shape": shape, "columns": columns}
    if "timestamp" in df.columns and not df.empty:
        extra["ts_min"] = str(df["timestamp"].min())
        extra["ts_max"] = str(df["timestamp"].max())
    if "value" in df.columns and not df.empty:
        extra["value_min"] = float(pd.to_numeric(df["value"], errors="coerce").min())
        extra["value_max"] = float(pd.to_numeric(df["value"], errors="coerce").max())
    logger.log(level, "DataFrame[%s] | %s", name, _safe_preview(extra, max_len=500))
    if preview_rows > 0 and not df.empty:
        logger.log(
            level,
            "DataFrame[%s] preview=%s",
            name,
            _safe_preview(df.head(preview_rows).to_dict(orient="records"), max_len=700),
        )


class StageTimer:
    def __init__(self, logger: logging.Logger, stage_name: str) -> None:
        self.logger = logger
        self.stage_name = stage_name
        self.started_at: Optional[float] = None
        self.started_dt: Optional[datetime] = None

    def __enter__(self) -> "StageTimer":
        self.started_at = time.monotonic()
        self.started_dt = datetime.utcnow()
        self.logger.info("STAGE START | %s | utc=%s", self.stage_name, self.started_dt.isoformat())
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = 0.0
        if self.started_at is not None:
            elapsed = time.monotonic() - self.started_at
        if exc is None:
            self.logger.info("STAGE END | %s | elapsed_sec=%.3f | status=ok", self.stage_name, elapsed)
            return False
        self.logger.exception(
            "STAGE END | %s | elapsed_sec=%.3f | status=error | error=%s",
            self.stage_name,
            elapsed,
            exc,
        )
        return False
