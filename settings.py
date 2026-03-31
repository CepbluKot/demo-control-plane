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
    CONTROL_PLANE_PROM_METRICS_STEP: str = Field(
        default="1m",
        validation_alias=AliasChoices(
            "CONTROL_PLANE_PROM_METRICS_STEP",
            "CONTROL_PLANE_PROM_STEP",
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

    # ClickHouse metrics source
    CONTROL_PLANE_CLICKHOUSE_METRICS_HOST: str = ""
    CONTROL_PLANE_CLICKHOUSE_METRICS_PORT: int = 8123
    CONTROL_PLANE_CLICKHOUSE_METRICS_USERNAME: str = ""
    CONTROL_PLANE_CLICKHOUSE_METRICS_PASSWORD: str = ""
    CONTROL_PLANE_CLICKHOUSE_METRICS_SECURE: bool = False
    CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY: str = ""
    CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST: str = ""
    CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PORT: int = 8123
    CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_USERNAME: str = ""
    CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PASSWORD: str = ""
    CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_SECURE: bool = False
    CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_QUERY: str = ""

    # ClickHouse logs source (for my_summarizer)
    CONTROL_PLANE_LOGS_CLICKHOUSE_HOST: str = "localhost"
    CONTROL_PLANE_LOGS_CLICKHOUSE_PORT: int = 8123
    CONTROL_PLANE_LOGS_CLICKHOUSE_USERNAME: str = ""
    CONTROL_PLANE_LOGS_CLICKHOUSE_PASSWORD: str = ""
    CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY: str = ""
    CONTROL_PLANE_LOGS_TIMESTAMP_COLUMN: str = "timestamp"
    CONTROL_PLANE_LOGS_PAGE_LIMIT: int = 1000
    CONTROL_PLANE_LOGS_FETCH_MODE: str = "time_window"
    CONTROL_PLANE_LOGS_TAIL_LIMIT: int = 1000
    CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_SQL: str = ""
    CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_METRICS_SQL: str = ""
    CONTROL_PLANE_UI_LOGS_SUMMARY_BATCH_SIZE: int = 200
    CONTROL_PLANE_UI_LOGS_SUMMARY_DB_BATCH_SIZE: int = 1000
    CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE: int = 200
    # 0 or negative disables per-cell truncation in prompts.
    CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS: int = 0
    # 0 or negative disables post-LLM truncation of map/reduce summaries.
    CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS: int = 0
    # 0 or negative disables local reduce prompt budget check.
    CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS: int = 0
    CONTROL_PLANE_UI_LOGS_SUMMARY_MAP_WORKERS: int = 1
    CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_RETRIES: int = -1
    CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_TIMEOUT: int = 600
    CONTROL_PLANE_HTTP_TIMEOUT_SECONDS: int = 600

    # Forecast metadata
    CONTROL_PLANE_FORECAST_SERVICE: str = "demo-service"
    CONTROL_PLANE_FORECAST_METRIC_NAME: str = "memory"
    CONTROL_PLANE_FORECAST_TYPE: str = "short"
    CONTROL_PLANE_PREDICTION_KIND: str = "forecast"

    # Anomaly detector
    CONTROL_PLANE_ANOMALY_DETECTOR: str = "rolling_iqr"
    CONTROL_PLANE_ANOMALY_ZSCORE: float = 3.0
    CONTROL_PLANE_ANOMALY_IQR_WINDOW: int = 60
    CONTROL_PLANE_ANOMALY_IQR_SCALE: float = 1.5
    CONTROL_PLANE_ANOMALY_IQR_MIN_PERIODS: int = 30
    CONTROL_PLANE_ANOMALY_PYOD_CONTAMINATION: float = 0.05
    CONTROL_PLANE_ANOMALY_PYOD_RANDOM_STATE: int = 42
    CONTROL_PLANE_ANOMALY_RUPTURES_PENALTY: float = 8.0
    CONTROL_PLANE_ANOMALY_RUPTURES_MODEL: str = "rbf"

    # Processing adapters
    CONTROL_PLANE_SUMMARY_LOG_CHARS: int = 200
    CONTROL_PLANE_SUMMARIZER_CALLABLE: str = ""
    CONTROL_PLANE_ALERT_CALLABLE: str = ""

    # Test mode synthetic generator
    CONTROL_PLANE_TEST_SEED: int = 42
    CONTROL_PLANE_TEST_SPIKE_RATE: float = 0.03
    CONTROL_PLANE_TEST_SPIKE_SCALE: float = 12.0
    CONTROL_PLANE_TEST_NOISE_SCALE: float = 1.5
    CONTROL_PLANE_TEST_SUMMARY_BATCHES: int = 4
    CONTROL_PLANE_TEST_LOGS_PER_BATCH: int = 120

    # LLM credentials
    OPENAI_API_BASE_DB: str = ""
    OPENAI_API_KEY_DB: str = ""
    LLM_MODEL_ID: str = "PNX.QWEN3 235b a22b instruct"
    CONTROL_PLANE_LLM_SYSTEM_PROMPT: str = ""
    CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT: str = ""
    CONTROL_PLANE_LLM_ANTI_HALLUCINATION_RULES: str = ""
    CONTROL_PLANE_LLM_MAP_PROMPT_TEMPLATE: str = ""
    CONTROL_PLANE_LLM_REDUCE_PROMPT_TEMPLATE: str = ""
    CONTROL_PLANE_LLM_CROSS_SOURCE_REDUCE_PROMPT_TEMPLATE: str = ""
    CONTROL_PLANE_LLM_FREEFORM_PROMPT_TEMPLATE: str = ""
    CONTROL_PLANE_LLM_UI_FINAL_REPORT_PROMPT_TEMPLATE: str = ""
    # Optional LLM generation cap (0 disables explicit max_tokens in request payload).
    CONTROL_PLANE_LLM_MAX_TOKENS: int = 0
    # If model stops with finish_reason=length/max_tokens, request continuation chunks.
    CONTROL_PLANE_LLM_CONTINUE_ON_LENGTH: bool = True
    # Max continuation chunks after the first response.
    CONTROL_PLANE_LLM_CONTINUE_MAX_ROUNDS: int = 12


settings = Settings()
