import os
from pathlib import Path
from typing import Optional

from settings import settings


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


_load_dotenv()

TOP_N_ANOMALIES = _as_int(os.getenv("CONTROL_PLANE_TOP_N_ANOMALIES"), 5)
ANALYZE_TOP_N_ANOMALIES = _as_int(os.getenv("CONTROL_PLANE_ANALYZE_TOP_N_ANOMALIES"), 1)
LOOPBACK_MINUTES = _as_int(os.getenv("CONTROL_PLANE_LOOPBACK_MINUTES"), 30)
DATA_LOOKBACK_MINUTES = _as_int(os.getenv("CONTROL_PLANE_DATA_LOOKBACK_MINUTES"), 60)
PREDICTION_LOOKAHEAD_MINUTES = _as_int(os.getenv("CONTROL_PLANE_PREDICTION_LOOKAHEAD_MINUTES"), 60)
METRICS_SOURCE = os.getenv("CONTROL_PLANE_METRICS_SOURCE", "prometheus").strip().lower()
# Prometheus range-step is technical API detail; keep it fixed internally.
PROM_STEP = "1m"
PROM_MAX_POINTS = _as_int(
    os.getenv(
        "CONTROL_PLANE_PROM_METRICS_MAX_POINTS",
        os.getenv("CONTROL_PLANE_PROM_MAX_POINTS"),
    ),
    11000,
)
PROM_SERIES_INDEX = _as_int(
    os.getenv(
        "CONTROL_PLANE_PROM_METRICS_SERIES_INDEX",
        os.getenv("CONTROL_PLANE_PROM_SERIES_INDEX"),
    ),
    0,
)
PROM_QUERY = os.getenv(
    "CONTROL_PLANE_PROM_METRICS_QUERY",
    os.getenv(
        "CONTROL_PLANE_PROM_QUERY",
    ),
)
if not PROM_QUERY:
    PROM_QUERY = getattr(
        settings,
        "PROM_QUERY",
        'sum(container_memory_working_set_bytes{image!="", '
        'container="airflow-worker", node="ndp-v01wnl-n19"})',
    )
PROM_URL = os.getenv(
    "CONTROL_PLANE_PROM_METRICS_URL",
    os.getenv(
        "CONTROL_PLANE_PROM_URL",
    ),
)
if not PROM_URL:
    PROM_URL = getattr(settings, "PROM_URL", "https://prom-ndp-v01.ndp-psi.int.gazprombank.ru")
PROM_USERNAME = os.getenv(
    "CONTROL_PLANE_PROM_METRICS_USERNAME",
    os.getenv(
        "CONTROL_PLANE_PROM_USERNAME",
    ),
)
if PROM_USERNAME is None or PROM_USERNAME == "":
    PROM_USERNAME = getattr(settings, "PROM_USERNAME", "ndp-monitor")
PROM_PASSWORD = os.getenv(
    "CONTROL_PLANE_PROM_METRICS_PASSWORD",
    os.getenv(
        "CONTROL_PLANE_PROM_PASSWORD",
    ),
)
if PROM_PASSWORD is None:
    PROM_PASSWORD = getattr(settings, "PROM_PASSWORD", "")
PROM_DISABLE_SSL = _as_bool(
    os.getenv(
        "CONTROL_PLANE_PROM_METRICS_DISABLE_SSL",
        os.getenv("CONTROL_PLANE_PROM_DISABLE_SSL"),
    ),
    bool(getattr(settings, "PROM_DISABLE_SSL", True)),
)

CLICKHOUSE_METRICS_QUERY = os.getenv(
    "CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY",
    os.getenv("CONTROL_PLANE_CLICKHOUSE_QUERY", ""),
)
CLICKHOUSE_METRICS_HOST = os.getenv(
    "CONTROL_PLANE_CLICKHOUSE_METRICS_HOST",
    os.getenv("CONTROL_PLANE_CLICKHOUSE_HOST", ""),
)
CLICKHOUSE_METRICS_PORT = _as_int(
    os.getenv(
        "CONTROL_PLANE_CLICKHOUSE_METRICS_PORT",
        os.getenv("CONTROL_PLANE_CLICKHOUSE_PORT"),
    ),
    8123,
)
CLICKHOUSE_METRICS_USERNAME = os.getenv(
    "CONTROL_PLANE_CLICKHOUSE_METRICS_USERNAME",
    os.getenv("CONTROL_PLANE_CLICKHOUSE_USERNAME", ""),
)
CLICKHOUSE_METRICS_PASSWORD = os.getenv(
    "CONTROL_PLANE_CLICKHOUSE_METRICS_PASSWORD",
    os.getenv("CONTROL_PLANE_CLICKHOUSE_PASSWORD", ""),
)
CLICKHOUSE_METRICS_DATABASE = os.getenv(
    "CONTROL_PLANE_CLICKHOUSE_METRICS_DATABASE",
    os.getenv("CONTROL_PLANE_CLICKHOUSE_DATABASE", ""),
)
CLICKHOUSE_METRICS_SECURE = _as_bool(
    os.getenv(
        "CONTROL_PLANE_CLICKHOUSE_METRICS_SECURE",
        os.getenv("CONTROL_PLANE_CLICKHOUSE_SECURE"),
    ),
    False,
)

FORECAST_SERVICE = os.getenv(
    "CONTROL_PLANE_FORECAST_SERVICE",
    getattr(settings, "FORECAST_SERVICE", "airflow-test-v1"),
)
FORECAST_METRIC_NAME = os.getenv(
    "CONTROL_PLANE_FORECAST_METRIC_NAME",
    getattr(settings, "FORECAST_METRIC_NAME", "memory"),
)
FORECAST_TYPE = os.getenv(
    "CONTROL_PLANE_FORECAST_TYPE",
    getattr(settings, "FORECAST_TYPE", "short"),
)
PREDICTION_KIND = os.getenv(
    "CONTROL_PLANE_PREDICTION_KIND",
    getattr(settings, "PREDICTION_KIND", "forecast"),
)

ANOMALY_ZSCORE = _as_float(os.getenv("CONTROL_PLANE_ANOMALY_ZSCORE"), 3.5)
ANOMALY_IQR_WINDOW = _as_int(os.getenv("CONTROL_PLANE_ANOMALY_IQR_WINDOW"), 60)
ANOMALY_IQR_SCALE = _as_float(os.getenv("CONTROL_PLANE_ANOMALY_IQR_SCALE"), 1.5)
ANOMALY_IQR_MIN_PERIODS = _as_int(os.getenv("CONTROL_PLANE_ANOMALY_IQR_MIN_PERIODS"), 30)
ANOMALY_DETECTOR = os.getenv("CONTROL_PLANE_ANOMALY_DETECTOR", "rolling_iqr")

PROCESS_ALERTS = _as_bool(os.getenv("CONTROL_PLANE_PROCESS_ALERTS"), True)
SUMMARY_LOG_CHARS = _as_int(os.getenv("CONTROL_PLANE_SUMMARY_LOG_CHARS"), 200)
SUMMARIZER_CALLABLE = os.getenv("CONTROL_PLANE_SUMMARIZER_CALLABLE", "").strip()
ALERT_CALLABLE = os.getenv("CONTROL_PLANE_ALERT_CALLABLE", "").strip()
LOG_LEVEL = os.getenv("CONTROL_PLANE_LOG_LEVEL", "DEBUG").upper()

ARTIFACTS_DIR = Path(os.getenv("CONTROL_PLANE_ARTIFACTS_DIR", "artifacts"))
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOGS_DIR = ARTIFACTS_DIR / "logs"

TEST_MODE = _as_bool(os.getenv("CONTROL_PLANE_TEST_MODE"), False)
TEST_MODE_SEED = _as_int(os.getenv("CONTROL_PLANE_TEST_SEED"), 42)
TEST_MODE_SPIKE_RATE = _as_float(os.getenv("CONTROL_PLANE_TEST_SPIKE_RATE"), 0.03)
TEST_MODE_SPIKE_SCALE = _as_float(os.getenv("CONTROL_PLANE_TEST_SPIKE_SCALE"), 12.0)
TEST_MODE_NOISE_SCALE = _as_float(os.getenv("CONTROL_PLANE_TEST_NOISE_SCALE"), 1.5)
