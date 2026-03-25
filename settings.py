"""
Centralized application settings loaded from environment/.env via Pydantic Settings.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core runtime
    CONTROL_PLANE_TEST_MODE: bool = False
    CONTROL_PLANE_PROCESS_ALERTS: bool = True
    CONTROL_PLANE_LOG_LEVEL: str = "DEBUG"
    CONTROL_PLANE_ARTIFACTS_DIR: str = "artifacts"

    # Data windows
    CONTROL_PLANE_TOP_N_ANOMALIES: int = 5
    CONTROL_PLANE_ANALYZE_TOP_N_ANOMALIES: int = 1
    CONTROL_PLANE_LOOPBACK_MINUTES: int = 30
    CONTROL_PLANE_DATA_LOOKBACK_MINUTES: int = 60
    CONTROL_PLANE_PREDICTION_LOOKAHEAD_MINUTES: int = 60

    # Metrics source
    CONTROL_PLANE_METRICS_SOURCE: str = "prometheus"

    # Prometheus metrics source (new names + legacy aliases)
    CONTROL_PLANE_PROM_METRICS_URL: str = Field(
        default="http://localhost:9090",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_PROM_METRICS_URL",
            "CONTROL_PLANE_PROM_URL",
        ),
    )
    CONTROL_PLANE_PROM_METRICS_USERNAME: str = Field(
        default="test-user",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_PROM_METRICS_USERNAME",
            "CONTROL_PLANE_PROM_USERNAME",
        ),
    )
    CONTROL_PLANE_PROM_METRICS_PASSWORD: str = Field(
        default="test-password",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_PROM_METRICS_PASSWORD",
            "CONTROL_PLANE_PROM_PASSWORD",
        ),
    )
    CONTROL_PLANE_PROM_METRICS_DISABLE_SSL: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "CONTROL_PLANE_PROM_METRICS_DISABLE_SSL",
            "CONTROL_PLANE_PROM_DISABLE_SSL",
        ),
    )
    CONTROL_PLANE_PROM_METRICS_QUERY: str = Field(
        default=(
            'sum(container_memory_working_set_bytes{image!="", '
            'container="airflow-worker", node="ndp-v01wnl-n19"})'
        ),
        validation_alias=AliasChoices(
            "CONTROL_PLANE_PROM_METRICS_QUERY",
            "CONTROL_PLANE_PROM_QUERY",
        ),
    )
    CONTROL_PLANE_PROM_METRICS_MAX_POINTS: int = Field(
        default=11000,
        validation_alias=AliasChoices(
            "CONTROL_PLANE_PROM_METRICS_MAX_POINTS",
            "CONTROL_PLANE_PROM_MAX_POINTS",
        ),
    )
    CONTROL_PLANE_PROM_METRICS_SERIES_INDEX: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "CONTROL_PLANE_PROM_METRICS_SERIES_INDEX",
            "CONTROL_PLANE_PROM_SERIES_INDEX",
        ),
    )

    # ClickHouse metrics source (new names + legacy aliases)
    CONTROL_PLANE_CLICKHOUSE_METRICS_HOST: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_CLICKHOUSE_METRICS_HOST",
            "CONTROL_PLANE_CLICKHOUSE_HOST",
        ),
    )
    CONTROL_PLANE_CLICKHOUSE_METRICS_PORT: int = Field(
        default=8123,
        validation_alias=AliasChoices(
            "CONTROL_PLANE_CLICKHOUSE_METRICS_PORT",
            "CONTROL_PLANE_CLICKHOUSE_PORT",
        ),
    )
    CONTROL_PLANE_CLICKHOUSE_METRICS_USERNAME: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_CLICKHOUSE_METRICS_USERNAME",
            "CONTROL_PLANE_CLICKHOUSE_USERNAME",
        ),
    )
    CONTROL_PLANE_CLICKHOUSE_METRICS_PASSWORD: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_CLICKHOUSE_METRICS_PASSWORD",
            "CONTROL_PLANE_CLICKHOUSE_PASSWORD",
        ),
    )
    CONTROL_PLANE_CLICKHOUSE_METRICS_SECURE: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "CONTROL_PLANE_CLICKHOUSE_METRICS_SECURE",
            "CONTROL_PLANE_CLICKHOUSE_SECURE",
        ),
    )
    CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY",
            "CONTROL_PLANE_CLICKHOUSE_QUERY",
        ),
    )

    # Logs source (for my_summarizer)
    CONTROL_PLANE_LOGS_CH_HOST: str = "localhost"
    CONTROL_PLANE_LOGS_CH_PORT: int = 8123
    CONTROL_PLANE_LOGS_CH_USERNAME: str = ""
    CONTROL_PLANE_LOGS_CH_PASSWORD: str = ""
    CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY: str = Field(
        default="",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY",
            "CONTROL_PLANE_LOGS_QUERY",
        ),
    )
    CONTROL_PLANE_LOGS_PAGE_LIMIT: int = 1000

    # Forecast metadata
    CONTROL_PLANE_FORECAST_SERVICE: str = "demo-service"
    CONTROL_PLANE_FORECAST_METRIC_NAME: str = "memory"
    CONTROL_PLANE_FORECAST_TYPE: str = "short"
    CONTROL_PLANE_PREDICTION_KIND: str = "forecast"

    # Anomaly detector
    CONTROL_PLANE_ANOMALY_DETECTOR: str = "rolling_iqr"
    CONTROL_PLANE_ANOMALY_ZSCORE: float = 3.5
    CONTROL_PLANE_ANOMALY_IQR_WINDOW: int = 60
    CONTROL_PLANE_ANOMALY_IQR_SCALE: float = 1.5
    CONTROL_PLANE_ANOMALY_IQR_MIN_PERIODS: int = 30

    # Processing adapters
    CONTROL_PLANE_SUMMARY_LOG_CHARS: int = 200
    CONTROL_PLANE_SUMMARIZER_CALLABLE: str = ""
    CONTROL_PLANE_ALERT_CALLABLE: str = ""

    # Test mode synthetic generator
    CONTROL_PLANE_TEST_SEED: int = 42
    CONTROL_PLANE_TEST_SPIKE_RATE: float = 0.03
    CONTROL_PLANE_TEST_SPIKE_SCALE: float = 12.0
    CONTROL_PLANE_TEST_NOISE_SCALE: float = 1.5

    # Predictions DB
    METRICS_PREDICTIONS_DATABASE: str = "default"

    # LLM credentials
    OPENAI_API_BASE_DB: str = ""
    OPENAI_API_KEY_DB: str = ""
    LLM_MODEL_ID: str = "PNX.QWEN3 235b a22b instruct"


settings = Settings()
