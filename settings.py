"""
Test settings for control plane demo.
Edit values here or override via environment variables where relevant.
"""

import os


class Settings:
    # Prometheus (test/demo values)
    PROM_URL = "http://localhost:9090"
    PROM_USERNAME = "test-user"
    PROM_PASSWORD = "test-password"
    PROM_DISABLE_SSL = True
    PROM_QUERY = (
        'sum(container_memory_working_set_bytes{image!="", '
        'container="airflow-worker", node="ndp-v01wnl-n19"})'
    )

    # Forecast metadata (test/demo values)
    FORECAST_SERVICE = "demo-service"
    FORECAST_METRIC_NAME = "memory"
    FORECAST_TYPE = "short"
    PREDICTION_KIND = "forecast"

    # Predictions DB (ClickHouse / sqlalchemy_stuff)
    TGT_DATABASE_HOST = os.getenv("TGT_DATABASE_HOST", "localhost")
    TGT_DATABASE_PORT = int(os.getenv("TGT_DATABASE_PORT", "8123"))
    TGT_DATABASE_USERNAME = os.getenv("TGT_DATABASE_USERNAME", "")
    TGT_DATABASE_PASSWORD = os.getenv("TGT_DATABASE_PASSWORD", "")
    METRICS_PREDICTIONS_DATABASE = os.getenv(
        "METRICS_PREDICTIONS_DATABASE",
        "default",
    )

    # LLM credentials / endpoint
    OPENAI_API_BASE_DB = os.getenv("OPENAI_API_BASE_DB", "")
    OPENAI_API_KEY_DB = os.getenv("OPENAI_API_KEY_DB", "")
    LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "PNX.QWEN3 235b a22b instruct")


settings = Settings()
