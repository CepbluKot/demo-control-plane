import logging
from datetime import datetime, timezone
from typing import Set

import pandas as pd

from settings import settings

from .trace import log_dataframe, log_event

logger = logging.getLogger(__name__)
METRICS_FORECAST_TABLE = "metrics_forecast"


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _escape_sql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def _to_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat()
    return value.astimezone(timezone.utc).isoformat()


def _query_metrics_df(query: str) -> pd.DataFrame:
    try:
        import clickhouse_connect
    except Exception as exc:
        raise ImportError("Для чтения предсказаний из ClickHouse нужен пакет clickhouse-connect") from exc

    host = str(settings.CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST).strip()
    if not host:
        raise ValueError(
            "Укажи CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST в .env "
            "для чтения предсказаний из ClickHouse"
        )

    client = clickhouse_connect.get_client(
        host=host,
        port=int(settings.CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PORT),
        username=str(settings.CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_USERNAME).strip() or None,
        password=str(settings.CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PASSWORD).strip() or None,
        secure=bool(settings.CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_SECURE),
    )
    try:
        return client.query_df(query)
    finally:
        try:
            client.close()
        except Exception:
            logger.warning("ClickHouse client close failed for predictions query")


def _load_metrics_forecast_columns() -> Set[str]:
    describe_df = _query_metrics_df(f"DESCRIBE TABLE {METRICS_FORECAST_TABLE}")
    if describe_df.empty:
        return set()
    name_col = "name" if "name" in describe_df.columns else str(describe_df.columns[0])
    return {str(value) for value in describe_df[name_col].dropna().tolist()}


def _render_predictions_query(
    query_template: str,
    *,
    service: str,
    metric_name: str,
    forecast_type: str,
    prediction_kind: str,
    start_time: datetime,
    end_time: datetime,
) -> str:
    start_iso = _to_utc_iso(start_time)
    end_iso = _to_utc_iso(end_time)
    params = _SafeFormatDict(
        table=METRICS_FORECAST_TABLE,
        service=_escape_sql_string(service),
        metric_name=_escape_sql_string(metric_name),
        forecast_type=_escape_sql_string(forecast_type),
        prediction_kind=_escape_sql_string(prediction_kind),
        start=_escape_sql_string(start_iso),
        end=_escape_sql_string(end_iso),
        start_iso=_escape_sql_string(start_iso),
        end_iso=_escape_sql_string(end_iso),
        start_ts=int(start_time.replace(tzinfo=timezone.utc).timestamp())
        if start_time.tzinfo is None
        else int(start_time.astimezone(timezone.utc).timestamp()),
        end_ts=int(end_time.replace(tzinfo=timezone.utc).timestamp())
        if end_time.tzinfo is None
        else int(end_time.astimezone(timezone.utc).timestamp()),
    )
    return query_template.strip().rstrip(";").format_map(params)


def _normalize_predictions_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("DB predictions fetch done: no rows")
        log_dataframe(logger, "predictions_df", df)
        return pd.DataFrame(columns=["timestamp", "predicted"])

    if "predicted" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "predicted"})

    if "timestamp" not in df.columns or "predicted" not in df.columns:
        raise ValueError("Predictions query должен вернуть колонки timestamp и predicted (или value)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    out = df.dropna(subset=["timestamp", "predicted"]).sort_values("timestamp").reset_index(drop=True)
    log_dataframe(logger, "predictions_df", out)
    return out


def fetch_predictions_from_db(
    service: str,
    metric_name: str,
    start_time: datetime,
    end_time: datetime,
    forecast_type: str = "short",
    prediction_kind: str = "forecast",
) -> pd.DataFrame:
    log_event(
        logger,
        "fetch_predictions_from_db.start",
        service=service,
        metric_name=metric_name,
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        forecast_type=forecast_type,
        prediction_kind=prediction_kind,
    )
    logger.info(
        "DB predictions fetch start: service=%s, metric=%s, forecast_type=%s, kind=%s",
        service,
        metric_name,
        forecast_type,
        prediction_kind,
    )

    custom_query_template = str(settings.CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_QUERY).strip()
    if custom_query_template:
        custom_query = _render_predictions_query(
            custom_query_template,
            service=service,
            metric_name=metric_name,
            forecast_type=forecast_type,
            prediction_kind=prediction_kind,
            start_time=start_time,
            end_time=end_time,
        )
        log_event(
            logger,
            "fetch_predictions_from_db.custom_query",
            query_preview=custom_query[:800],
        )
        df = _query_metrics_df(custom_query)
        log_event(logger, "fetch_predictions_from_db.rows_loaded", rows=len(df), mode="custom")
        out = _normalize_predictions_df(df)
        logger.info("DB predictions fetch done: mode=custom, points=%s", len(out))
        return out

    log_event(logger, "fetch_predictions_from_db.table", table=METRICS_FORECAST_TABLE)
    available_columns = _load_metrics_forecast_columns()
    log_event(
        logger,
        "fetch_predictions_from_db.columns_detected",
        columns=sorted(available_columns),
    )

    where_parts = [
        f"service = '{_escape_sql_string(service)}'",
        f"metric_name = '{_escape_sql_string(metric_name)}'",
    ]
    if "prediction_kind" in available_columns:
        where_parts.append(f"prediction_kind = '{_escape_sql_string(prediction_kind)}'")
    if "forecast_type" in available_columns:
        where_parts.append(f"forecast_type = '{_escape_sql_string(forecast_type)}'")
    where_sql = " AND ".join(where_parts)
    log_event(logger, "fetch_predictions_from_db.filters_built", filters=where_parts)

    latest_gen_query = (
        f"SELECT max(generated_at) AS generated_at "
        f"FROM {METRICS_FORECAST_TABLE} "
        f"WHERE {where_sql}"
    )
    latest_df = _query_metrics_df(latest_gen_query)
    latest_generated_at = None
    if not latest_df.empty and "generated_at" in latest_df.columns:
        latest_generated_at = latest_df["generated_at"].iloc[0]
    if latest_generated_at is None or pd.isna(latest_generated_at):
        raise ValueError("Не найдено предсказаний в БД для заданных фильтров")
    log_event(logger, "fetch_predictions_from_db.latest_generated_at", value=str(latest_generated_at))

    start_iso = _escape_sql_string(_to_utc_iso(start_time))
    end_iso = _escape_sql_string(_to_utc_iso(end_time))
    predictions_query = (
        f"SELECT timestamp, value AS predicted "
        f"FROM {METRICS_FORECAST_TABLE} "
        f"WHERE {where_sql} "
        f"  AND generated_at = ("
        f"      SELECT max(generated_at) "
        f"      FROM {METRICS_FORECAST_TABLE} "
        f"      WHERE {where_sql}"
        f"  ) "
        f"  AND timestamp >= parseDateTimeBestEffort('{start_iso}') "
        f"  AND timestamp <= parseDateTimeBestEffort('{end_iso}') "
        f"ORDER BY timestamp"
    )
    query_preview = predictions_query[:800]
    log_event(logger, "fetch_predictions_from_db.query", query_preview=query_preview)
    df = _query_metrics_df(predictions_query)
    log_event(logger, "fetch_predictions_from_db.rows_loaded", rows=len(df))
    df = _normalize_predictions_df(df)
    logger.info(
        "DB predictions fetch done: generated_at=%s, points=%s",
        latest_generated_at,
        len(df),
    )
    return df
