from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.python import get_current_context


DAG_ID = "summary_llm_connectivity_probe"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 8
MAX_ERROR_BODY_CHARS = 4000
MAX_RESPONSE_BODY_BYTES = 1024 * 1024


def _variable_or_env(name: str, *, fallback_env: str | None = None, default: str | None = None) -> str | None:
    value = Variable.get(name, default_var=None)
    if value:
        return value

    env_value = os.getenv(name)
    if env_value:
        return env_value

    if fallback_env:
        fallback_value = os.getenv(fallback_env)
        if fallback_value:
            return fallback_value

    return default


def _first_config_value(names: tuple[str, ...], *, default: str | None = None) -> str | None:
    for name in names:
        value = _variable_or_env(name)
        if value:
            return value.strip()
    return default


def _required_config_value(names: tuple[str, ...]) -> str:
    value = _first_config_value(names)
    if not value:
        readable_names = ", ".join(names)
        raise RuntimeError(f"Missing Airflow Variable/env. Tried: {readable_names}")
    return value


def _build_chat_completions_url(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if not base:
        raise RuntimeError("LLM API base URL is empty")
    if base.endswith("/chat/completions"):
        return base
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def _read_http_error_body(error: urllib.error.HTTPError) -> str:
    body = error.read(MAX_RESPONSE_BODY_BYTES).decode("utf-8", errors="replace")
    return body[:MAX_ERROR_BODY_CHARS]


def _request_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are handling a connectivity probe. Reply briefly.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "airflow-summary-llm-connectivity-probe/1.0",
        },
    )

    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read(MAX_RESPONSE_BODY_BYTES)
            response_body = raw_body.decode("utf-8", errors="replace")
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            return {
                "status": response.status,
                "elapsed_ms": elapsed_ms,
                "body": json.loads(response_body),
            }
    except urllib.error.HTTPError as error:
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        body = _read_http_error_body(error)
        raise RuntimeError(
            f"LLM probe failed with HTTP {error.code} after {elapsed_ms} ms. "
            f"Response body: {body}"
        ) from error
    except urllib.error.URLError as error:
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        raise RuntimeError(f"LLM probe failed after {elapsed_ms} ms: {error}") from error
    except json.JSONDecodeError as error:
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        raise RuntimeError(f"LLM probe returned non-JSON response after {elapsed_ms} ms") from error


def _extract_message_content(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM probe response does not contain choices")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("LLM probe response does not contain choices[0].message")

    content = message.get("content")
    if content is None:
        raise RuntimeError("LLM probe response message has empty content")
    return str(content)


@dag(
    dag_id=DAG_ID,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["summary", "llm", "connectivity"],
    default_args={"owner": "airflow", "retries": 0},
)
def summary_llm_connectivity_probe():
    @task(task_id="probe_llm_service")
    def probe_llm_service() -> dict[str, Any]:
        context = get_current_context()
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run and dag_run.conf else {}

        api_base = str(
            conf.get("api_base")
            or _required_config_value(
                (
                    "SUMMARY_BACKEND_OPENAI_API_BASE",
                    "OPENAI_API_BASE_DB",
                    "OPENAI_BASE_URL",
                    "LLM_API_BASE",
                    "OPENAI_API_BASE",
                    "API_BASE",
                )
            )
        )
        api_key = _required_config_value(
            (
                "SUMMARY_BACKEND_OPENAI_API_KEY",
                "OPENAI_API_KEY_DB",
                "LLM_API_KEY",
                "OPENAI_API_KEY",
                "API_KEY",
            )
        )
        model = str(
            conf.get("model")
            or _required_config_value(
                (
                    "SUMMARY_BACKEND_LLM_MODEL",
                    "LLM_MODEL_ID",
                    "OPENAI_MODEL",
                    "OPENAI_BIG_MODEL",
                    "LLM_MODEL",
                    "MODEL_ID",
                    "MODEL",
                )
            )
        )
        timeout_seconds = float(
            conf.get("timeout_seconds")
            or _first_config_value(
                ("SUMMARY_BACKEND_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT_SECONDS"),
                default=str(DEFAULT_TIMEOUT_SECONDS),
            )
            or DEFAULT_TIMEOUT_SECONDS
        )
        max_tokens = int(conf.get("max_tokens") or DEFAULT_MAX_TOKENS)
        prompt = str(conf.get("prompt") or "Reply with OK.")

        endpoint = _build_chat_completions_url(api_base)
        print(f"Probing LLM endpoint: {endpoint}")
        print(f"Model: {model}")
        print(f"Timeout seconds: {timeout_seconds}")

        result = _request_chat_completion(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        response_body = result["body"]
        content = _extract_message_content(response_body)
        usage = response_body.get("usage", {})

        print(f"LLM probe succeeded: status={result['status']} elapsed_ms={result['elapsed_ms']}")
        print(f"LLM response content: {content[:500]}")
        print(f"LLM usage: {json.dumps(usage, ensure_ascii=False)}")

        return {
            "ok": True,
            "status": result["status"],
            "elapsed_ms": result["elapsed_ms"],
            "model": model,
            "content_preview": content[:500],
            "usage": usage,
        }

    probe_llm_service()


summary_llm_connectivity_probe()
