import logging

import pandas as pd

from .trace import log_event

logger = logging.getLogger(__name__)


def to_iso_z(ts: pd.Timestamp) -> str:
    log_event(logger, "to_iso_z.start", ts=str(ts))
    if pd.isna(ts):
        log_event(logger, "to_iso_z.empty")
        return ""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    result = ts.isoformat().replace("+00:00", "Z")
    log_event(logger, "to_iso_z.done", result=result)
    return result
