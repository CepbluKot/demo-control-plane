from __future__ import annotations

from clickhouse_sqlalchemy import engines, types
from sqlalchemy import Column, Float, String, func

from .engine import Base


class MetricsForecast(Base):
    __tablename__ = "metrics_forecast"

    timestamp = Column(
        types.DateTime64(precision=6, timezone="UTC"),
        nullable=False,
        primary_key=True,
    )
    service = Column(String(256), nullable=False, primary_key=True)
    metric_name = Column(String(256), nullable=False, primary_key=True)
    generated_at = Column(
        types.DateTime64(precision=6, timezone="UTC"),
        nullable=False,
        primary_key=True,
    )
    value = Column(Float, nullable=False)
    model_version = Column(String(64), nullable=False, default="unknown")

    __table_args__ = (
        engines.MergeTree(
            partition_by=func.toYYYYMM(timestamp),
            order_by=(service, metric_name, timestamp, generated_at),
            ttl=func.toDateTime(generated_at + func.toIntervalDay(30)),
        ),
    )
