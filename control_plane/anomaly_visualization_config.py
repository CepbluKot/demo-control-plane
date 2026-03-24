VISUALIZATION_CONFIG = {
    "output_dir": "artifacts/plots",
    "datetime_format": "%Y%m%d_%H%M%S",
    "main_plot": {
        "figsize": (14, 6),
        "dpi": 150,
        "filename_prefix": "main_anomalies",
    },
    "anomaly_plot": {
        "figsize": (12, 6),
        "dpi": 150,
        "filename_prefix": "anomaly_detail",
        "loopback_minutes": 30,
    },
}
