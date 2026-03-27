import logging
from typing import Any, Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from settings import settings
from .trace import log_event

logger = logging.getLogger(__name__)


def create_session_with_retries() -> requests.Session:
    """Create requests session with retry strategy"""
    log_event(logger, "create_session_with_retries.start")
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    log_event(logger, "create_session_with_retries.done")
    return session


def fetch_anomalies_from_api(query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch anomalies from anomaly detection API.
    Args:
        query: PromQL query to analyze
    Returns:
        Tuple of (list of anomalies, full API response)
    """
    url = "http://localhost:8001/analyze/"
    log_event(logger, "fetch_anomalies_from_api.start", query=query, url=url)
    try:
        session = create_session_with_retries()
        response = session.post(
            url=url,
            params={"query": query},
            headers={"accept": "application/json"},
            json={},
            timeout=max(int(getattr(settings, "CONTROL_PLANE_HTTP_TIMEOUT_SECONDS", 600)), 1),
        )
        response.raise_for_status()
        log_event(logger, "fetch_anomalies_from_api.response", status_code=response.status_code)
        data = response.json()
        # Extract anomalies from response
        anomalies = []
        if "data" in data and "anomalies" in data["data"]:
            for anomaly in data["data"]["anomalies"]:
                # Ensure timestamp is in ISO format with timezone
                timestamp = anomaly["timestamp"]
                if not timestamp.endswith("Z"):
                    timestamp = f"{timestamp}Z"
                anomalies.append(
                    {
                        "timestamp": timestamp,
                        "value": anomaly["value"],
                        "predicted": anomaly["predicted"],
                        "residual": anomaly["residual"],
                        "is_anomaly": anomaly["is_anomaly"],
                    }
                )
        logger.info(f"Fetched {len(anomalies)} anomalies from API")
        log_event(logger, "fetch_anomalies_from_api.done", anomalies=len(anomalies))
        return anomalies, data
    except requests.exceptions.RequestException as exc:
        error_msg = f"Error fetching anomalies from API: {str(exc)}"
        logger.exception(error_msg)
        raise
    except Exception as exc:
        error_msg = f"Error parsing API response: {str(exc)}"
        logger.exception(error_msg)
        raise


def build_api_response(
    merged_df,
    predictions_df,
    anomalies_df,
) -> Dict[str, Any]:
    log_event(
        logger,
        "build_api_response.start",
        merged_rows=len(merged_df),
        prediction_rows=len(predictions_df),
        anomaly_rows=len(anomalies_df),
    )
    response = {
        "status": "ok",
        "data": {
            "merged_data": merged_df.to_dict(orient="records"),
            "predictions": predictions_df.to_dict(orient="records"),
            "anomalies": anomalies_df.to_dict(orient="records"),
        },
    }
    log_event(
        logger,
        "build_api_response.done",
        merged=len(response["data"]["merged_data"]),
        predictions=len(response["data"]["predictions"]),
        anomalies=len(response["data"]["anomalies"]),
    )
    return response
