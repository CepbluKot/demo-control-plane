from __future__ import annotations

from urllib.parse import quote_plus

from clickhouse_sqlalchemy import get_declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from settings import settings


def _read_setting(name: str, default: str | None = None) -> str | None:
    raw = getattr(settings, name, None)
    if raw is None or raw == "":
        return default
    return str(raw)


def _build_clickhouse_url(
    *,
    host: str,
    port: str,
    username: str,
    password: str,
    database: str = "default",
) -> str:
    encoded_user = quote_plus(username)
    encoded_pass = quote_plus(password)
    if encoded_user:
        return f"clickhouse://{encoded_user}:{encoded_pass}@{host}:{port}/{database}"
    return f"clickhouse://{host}:{port}/{database}"


# Metrics DB connection (used by MetricsForecast table + predictions reader)
metrics_host = _read_setting("CONTROL_PLANE_CLICKHOUSE_METRICS_HOST", default="localhost") or "localhost"
metrics_port = _read_setting("CONTROL_PLANE_CLICKHOUSE_METRICS_PORT", default="8123") or "8123"
metrics_username = _read_setting("CONTROL_PLANE_CLICKHOUSE_METRICS_USERNAME", default="") or ""
metrics_password = _read_setting("CONTROL_PLANE_CLICKHOUSE_METRICS_PASSWORD", default="") or ""
METRICS_CLICKHOUSE_URL = _build_clickhouse_url(
    host=metrics_host,
    port=metrics_port,
    username=metrics_username,
    password=metrics_password,
)


# Logs DB connection (separate session factory, separate host)
logs_host = _read_setting("CONTROL_PLANE_LOGS_CLICKHOUSE_HOST", default="localhost") or "localhost"
logs_port = _read_setting("CONTROL_PLANE_LOGS_CLICKHOUSE_PORT", default="8123") or "8123"
logs_username = _read_setting("CONTROL_PLANE_LOGS_CLICKHOUSE_USERNAME", default="") or ""
logs_password = _read_setting("CONTROL_PLANE_LOGS_CLICKHOUSE_PASSWORD", default="") or ""
LOGS_CLICKHOUSE_URL = _build_clickhouse_url(
    host=logs_host,
    port=logs_port,
    username=logs_username,
    password=logs_password,
)


metrics_engine = create_engine(METRICS_CLICKHOUSE_URL, echo=False)
logs_engine = create_engine(LOGS_CLICKHOUSE_URL, echo=False)

# Metrics ORM base: MetricsForecast maps strictly to metrics host connection.
Base = get_declarative_base()

# Explicit session factories
MetricsSession = sessionmaker(bind=metrics_engine)
LogsSession = sessionmaker(bind=logs_engine)
