"""Helpers for executing Dify workflows by API key."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings, get_settings

API_KEY_ENV_BY_WORKFLOW_ID: dict[str, str] = {
    "pa-inline-points-prediction-pipeline": "DIFY_API_KEY_PA_INLINE_POINTS_PREDICTION_PIPELINE",
    "pa-anomaly-detection-pipeline": "DIFY_API_KEY_PA_ANOMALY_DETECTION_PIPELINE",
    "pa-general-summary-generator": "DIFY_API_KEY_PA_GENERAL_SUMMARY_GENERATOR",
    "pa-control-plane-main-orchestrator": "DIFY_API_KEY_PA_CONTROL_PLANE_MAIN_ORCHESTRATOR",
    "pa-local-orchestrator": "DIFY_API_KEY_PA_LOCAL_ORCHESTRATOR",
}


def _workflow_token(workflow_id: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(workflow_id or "").strip()).upper()


def _extract_error_message(payload: Any, status_code: int) -> str:
    if isinstance(payload, dict):
        for key in ("message", "msg", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if isinstance(payload.get("detail"), list) and payload["detail"]:
            return json.dumps(payload["detail"], ensure_ascii=False)
    if isinstance(payload, str) and payload.strip():
        return payload
    return f"Dify request failed with status {status_code}."


@dataclass(frozen=True)
class DifyWorkflowRunResult:
    status: str
    workflow_run_id: str
    task_id: str
    outputs: dict[str, Any]
    raw: dict[str, Any]


class DifyWorkflowClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _read_secrets(self) -> dict[str, str]:
        path = Path(self.settings.dify_secrets_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items() if str(value).strip()}

    def resolve_api_key(self, workflow_id: str) -> str:
        env_name = API_KEY_ENV_BY_WORKFLOW_ID.get(workflow_id) or f"SUMMARY_BACKEND_DIFY_WORKFLOW__{_workflow_token(workflow_id)}__API_KEY"
        from_env = __import__("os").environ.get(env_name, "").strip()
        if from_env:
            return from_env
        return self._read_secrets().get(env_name, "").strip()

    def run_workflow(
        self,
        *,
        workflow_id: str,
        inputs: dict[str, Any],
        user: str,
        timeout_seconds: float | None = None,
    ) -> DifyWorkflowRunResult:
        api_key = self.resolve_api_key(workflow_id)
        if not api_key:
            raise ValueError(f"Dify API key is not configured for workflow '{workflow_id}'.")

        timeout = timeout_seconds if timeout_seconds is not None else self.settings.dify_timeout_seconds
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.settings.dify_api_base.rstrip('/')}/v1/workflows/run",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": inputs,
                    "response_mode": "blocking",
                    "user": user,
                },
            )

        try:
            payload = response.json() if response.text else {}
        except json.JSONDecodeError:
            payload = {"raw_text": response.text}

        if not response.is_success:
            raise RuntimeError(_extract_error_message(payload, response.status_code))

        data = payload.get("data") if isinstance(payload, dict) else {}
        outputs = data.get("outputs") if isinstance(data, dict) else {}
        return DifyWorkflowRunResult(
            status=str((data or {}).get("status") or ""),
            workflow_run_id=str((payload or {}).get("workflow_run_id") or ""),
            task_id=str((payload or {}).get("task_id") or ""),
            outputs=outputs if isinstance(outputs, dict) else {},
            raw=payload if isinstance(payload, dict) else {"raw_text": str(payload)},
        )
