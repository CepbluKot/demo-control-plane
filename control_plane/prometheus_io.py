import base64
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
from prometheus_api_client import PrometheusConnect

from .config import PROM_DISABLE_SSL, PROM_PASSWORD, PROM_URL, PROM_USERNAME
from .trace import log_dataframe, log_event

logger = logging.getLogger(__name__)


def _extract_secret(value: Any) -> Any:
    log_event(logger, "_extract_secret.start", has_method=hasattr(value, "get_secret_value"))
    if hasattr(value, "get_secret_value"):
        try:
            secret = value.get_secret_value()
            log_event(logger, "_extract_secret.done", extracted=True)
            return secret
        except Exception:
            logger.exception("_extract_secret failed to extract secret value")
            return value
    log_event(logger, "_extract_secret.done", extracted=False)
    return value


def get_prometheus_client() -> PrometheusConnect:
    url = PROM_URL
    if not url:
        raise ValueError("settings.PROM_URL обязателен")
    username = _extract_secret(PROM_USERNAME)
    password = _extract_secret(PROM_PASSWORD)
    disable_ssl = PROM_DISABLE_SSL
    logger.info(
        "Prometheus auth setup: url=%s, user_set=%s, pass_set=%s, disable_ssl=%s",
        url,
        bool(username),
        bool(password),
        disable_ssl,
    )
    headers = None
    if username is not None and password is not None:
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {"Authorization": f"Basic {encoded_credentials}"}
    return PrometheusConnect(url=url, disable_ssl=disable_ssl, headers=headers)


def parse_step_to_seconds(step: str) -> int:
    log_event(logger, "parse_step_to_seconds.start", step=step)
    step = str(step).strip()
    if step.endswith("s"):
        result = int(step[:-1])
        log_event(logger, "parse_step_to_seconds.done", result=result)
        return result
    if step.endswith("m"):
        result = int(step[:-1]) * 60
        log_event(logger, "parse_step_to_seconds.done", result=result)
        return result
    if step.endswith("h"):
        result = int(step[:-1]) * 3600
        log_event(logger, "parse_step_to_seconds.done", result=result)
        return result
    if step.endswith("d"):
        result = int(step[:-1]) * 86400
        log_event(logger, "parse_step_to_seconds.done", result=result)
        return result
    result = int(step)
    log_event(logger, "parse_step_to_seconds.done", result=result)
    return result


def calculate_max_range(step_seconds: int, max_points: int = 11000) -> int:
    safe_points = int(max_points * 0.9)
    result = safe_points * step_seconds
    log_event(
        logger,
        "calculate_max_range",
        step_seconds=step_seconds,
        max_points=max_points,
        safe_points=safe_points,
        result=result,
    )
    return result


def fetch_metric_in_batches(
    prom: PrometheusConnect,
    query: str,
    start_time: datetime,
    end_time: datetime,
    step: str,
    max_points: int = 11000,
) -> List[Dict[str, Any]]:
    log_event(
        logger,
        "fetch_metric_in_batches.start",
        query=query,
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        step=step,
        max_points=max_points,
    )
    step_seconds = parse_step_to_seconds(step)
    max_range_seconds = calculate_max_range(step_seconds, max_points)
    all_data: Dict[str, Dict[str, Any]] = {}
    current_start = start_time
    batch_index = 0

    while current_start < end_time:
        batch_index += 1
        current_end = min(current_start + timedelta(seconds=max_range_seconds), end_time)
        log_event(
            logger,
            "fetch_metric_in_batches.batch",
            batch_index=batch_index,
            batch_start=current_start.isoformat(),
            batch_end=current_end.isoformat(),
        )
        try:
            result = prom.custom_query_range(
                query=query, start_time=current_start, end_time=current_end, step=step
            )
            log_event(
                logger,
                "fetch_metric_in_batches.batch_result",
                batch_index=batch_index,
                series_count=len(result) if result else 0,
            )
            if result:
                for series in result:
                    series_id = str(sorted(series["metric"].items()))
                    if series_id not in all_data:
                        all_data[series_id] = {"metric": series["metric"], "values": []}
                    existing_timestamps = {ts for ts, _ in all_data[series_id]["values"]}
                    new_values = [
                        (ts, val) for ts, val in series["values"] if ts not in existing_timestamps
                    ]
                    all_data[series_id]["values"].extend(new_values)
        except Exception as exc:
            logger.exception("Prometheus batch error: %s", exc)
        current_start = current_end
        time.sleep(0.05)

    for series_id in all_data:
        all_data[series_id]["values"].sort(key=lambda x: x[0])
    out = list(all_data.values())
    log_event(
        logger,
        "fetch_metric_in_batches.done",
        batches=batch_index,
        series=len(out),
    )
    return out


def series_to_dataframe(series: Dict[str, Any]) -> pd.DataFrame:
    log_event(logger, "series_to_dataframe.start", values=len(series.get("values", [])))
    values = series["values"]
    df = pd.DataFrame(values, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    out = df.dropna().reset_index(drop=True)
    log_dataframe(logger, "prometheus_series_df", out)
    return out


def fetch_prometheus_df(
    query: str,
    start_time: datetime,
    end_time: datetime,
    step: str = "1m",
    max_points: int = 11000,
    series_index: int = 0,
) -> pd.DataFrame:
    logger.info(
        "Prometheus fetch start: query=%s, start=%s, end=%s, step=%s",
        query,
        start_time.isoformat(),
        end_time.isoformat(),
        step,
    )
    prom = get_prometheus_client()
    result = fetch_metric_in_batches(
        prom=prom,
        query=query,
        start_time=start_time,
        end_time=end_time,
        step=step,
        max_points=max_points,
    )
    if not result:
        raise ValueError("Нет данных из Prometheus по заданному запросу")
    if series_index >= len(result):
        raise IndexError(f"series_index={series_index} вне диапазона (серий: {len(result)})")
    df = series_to_dataframe(result[series_index])
    logger.info("Prometheus fetch done: points=%s", len(df))
    log_dataframe(logger, "prometheus_df", df)
    return df
