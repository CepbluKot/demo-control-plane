import logging
from datetime import datetime

import pandas as pd

from .trace import log_dataframe, log_event

logger = logging.getLogger(__name__)


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
    try:
        from sqlalchemy import func
        from sqlalchemy_stuff.engine import Session
        from sqlalchemy_stuff.tables import MetricsForecast
    except Exception as exc:
        logger.exception("DB predictions fetch import error")
        raise ImportError("sqlalchemy_stuff недоступен для чтения предсказаний из БД") from exc

    session = Session()
    try:
        filters = [
            MetricsForecast.service == service,
            MetricsForecast.metric_name == metric_name,
        ]
        if hasattr(MetricsForecast, "prediction_kind"):
            filters.append(MetricsForecast.prediction_kind == prediction_kind)
        if hasattr(MetricsForecast, "forecast_type"):
            filters.append(MetricsForecast.forecast_type == forecast_type)
        log_event(logger, "fetch_predictions_from_db.filters_built", filters_count=len(filters))
        latest_gen = session.query(func.max(MetricsForecast.generated_at)).filter(*filters).scalar()
        if latest_gen is None:
            raise ValueError("Не найдено предсказаний в БД для заданных фильтров")
        log_event(logger, "fetch_predictions_from_db.latest_generated_at", value=str(latest_gen))
        rows = (
            session.query(MetricsForecast.timestamp, MetricsForecast.value)
            .filter(*filters)
            .filter(MetricsForecast.generated_at == latest_gen)
            .filter(MetricsForecast.timestamp >= start_time)
            .filter(MetricsForecast.timestamp <= end_time)
            .all()
        )
        log_event(logger, "fetch_predictions_from_db.rows_loaded", rows=len(rows))
    finally:
        session.close()
        log_event(logger, "fetch_predictions_from_db.session_closed")

    df = pd.DataFrame(rows, columns=["timestamp", "predicted"])
    if df.empty:
        logger.warning("DB predictions fetch done: no rows")
        log_dataframe(logger, "predictions_df", df)
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "DB predictions fetch done: generated_at=%s, points=%s",
        latest_gen,
        len(df),
    )
    log_dataframe(logger, "predictions_df", df)
    return df
