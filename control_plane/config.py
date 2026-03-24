import os
from pathlib import Path

from settings import settings

TOP_N_ANOMALIES = 5
ANALYZE_TOP_N_ANOMALIES = 1
LOOPBACK_MINUTES = 30
DATA_LOOKBACK_MINUTES = 60
PREDICTION_LOOKAHEAD_MINUTES = 60
PROM_STEP = "1m"
PROM_MAX_POINTS = 11000
PROM_SERIES_INDEX = 0
PROM_QUERY = getattr(
    settings,
    "PROM_QUERY",
    'sum(container_memory_working_set_bytes{image!="", '
    'container="airflow-worker", node="ndp-v01wnl-n19"})',
)
PROM_URL = getattr(settings, "PROM_URL", "https://prom-ndp-v01.ndp-psi.int.gazprombank.ru")
PROM_USERNAME = getattr(settings, "PROM_USERNAME", "ndp-monitor")
PROM_PASSWORD = getattr(settings, "PROM_PASSWORD", "Adsa32423sa#asEDWQA1DSA")
PROM_DISABLE_SSL = bool(getattr(settings, "PROM_DISABLE_SSL", True))
FORECAST_SERVICE = getattr(settings, "FORECAST_SERVICE", "airflow-test-v1")
FORECAST_METRIC_NAME = getattr(settings, "FORECAST_METRIC_NAME", "memory")
FORECAST_TYPE = getattr(settings, "FORECAST_TYPE", "short")
PREDICTION_KIND = getattr(settings, "PREDICTION_KIND", "forecast")
ANOMALY_ZSCORE = 3.5
ANOMALY_IQR_WINDOW = 60
ANOMALY_IQR_SCALE = 1.5
ANOMALY_IQR_MIN_PERIODS = 30
ANOMALY_DETECTOR = os.getenv("CONTROL_PLANE_ANOMALY_DETECTOR", "rolling_iqr")
SUMMARY_LOG_CHARS = 200
LOG_LEVEL = os.getenv("CONTROL_PLANE_LOG_LEVEL", "DEBUG").upper()
ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOGS_DIR = ARTIFACTS_DIR / "logs"

TEST_MODE = os.getenv("CONTROL_PLANE_TEST_MODE", "0").lower() in {"1", "true", "yes", "y"}
TEST_MODE_SEED = int(os.getenv("CONTROL_PLANE_TEST_SEED", "42"))
TEST_MODE_SPIKE_RATE = float(os.getenv("CONTROL_PLANE_TEST_SPIKE_RATE", "0.03"))
TEST_MODE_SPIKE_SCALE = float(os.getenv("CONTROL_PLANE_TEST_SPIKE_SCALE", "12.0"))
TEST_MODE_NOISE_SCALE = float(os.getenv("CONTROL_PLANE_TEST_NOISE_SCALE", "1.5"))
