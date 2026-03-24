"""
Test settings for control plane demo.
Edit values here or override via environment variables where relevant.
"""


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


settings = Settings()
