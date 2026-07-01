"""Structured LLM client with JSON schema and file audit."""

from __future__ import annotations

import json
import re
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .audit import AuditWriter
from .config import Settings, get_settings, resolve_llm_model_option
from .errors import LlmPoolBusyError, classify_error
from .llm_pool import acquire_llm_pool_slot
from .logging_setup import get_logger
from .ports import AuditSink, SummaryStore
from .schemas import SummaryResult
from .text import estimate_tokens

logger = get_logger("llm_client")

TModel = TypeVar("TModel", bound=BaseModel)


class _ConnectivityProbePayload(BaseModel):
    ok: bool
    message: str


def _build_openai_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        return base
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


class StructuredLLMClient:
    def __init__(
        self,
        *,
        store: SummaryStore,
        settings: Settings | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.audit = audit or AuditWriter(self.settings)

    def call_summary(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        model: str | None = None,
    ) -> SummaryResult:
        return self.call_structured(
            job_id=job_id,
            node_id=node_id,
            stage=stage,
            system=system,
            user=user,
            model=model,
            response_model=SummaryResult,
        )

    def call_structured(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        model: str | None = None,
        response_model: type[TModel],
        response_schema: dict[str, Any] | None = None,
    ) -> TModel:
        selected_option = resolve_llm_model_option(self.settings, model)
        selected_model = selected_option.model.strip()
        if self.settings.dry_run:
            return self._call_dry_run(
                job_id=job_id,
                node_id=node_id,
                stage=stage,
                system=system,
                user=user,
                    model=selected_model,
                    response_model=response_model,
                    response_schema=response_schema,
                )

        last_exc: Exception | None = None
        for attempt in range(1, self.settings.llm_max_retries + 2):
            try:
                return self._call_openai(
                    job_id=job_id,
                    node_id=node_id,
                    stage=stage,
                    system=system,
                    user=user,
                    model=selected_model,
                    response_model=response_model,
                    response_schema=response_schema,
                    api_base=selected_option.api_base or self.settings.openai_api_base,
                    api_key=next(
                        (
                            profile.api_key
                            for profile in self.settings.llm_profiles
                            if profile.profile_id == selected_option.profile_id
                        ),
                        self.settings.openai_api_key,
                    ),
                    attempt=attempt,
                )
            except LlmPoolBusyError:
                raise
            except Exception as exc:
                last_exc = exc
                error_class = classify_error(exc)
                logger.warning(
                    "llm_call_failed | job_id=%s node_id=%s stage=%s attempt=%s error_class=%s error=%s",
                    job_id,
                    node_id,
                    stage,
                    attempt,
                    error_class,
                    exc,
                )
                if attempt > self.settings.llm_max_retries:
                    break
                time.sleep(self.settings.llm_retry_backoff_seconds * attempt)

        assert last_exc is not None
        raise last_exc

    def probe_connection(
        self,
        *,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        selected_option = resolve_llm_model_option(self.settings, model)
        selected_model = selected_option.model.strip()
        api_base = (selected_option.api_base or self.settings.openai_api_base).strip()
        api_key = next(
            (
                profile.api_key
                for profile in self.settings.llm_profiles
                if profile.profile_id == selected_option.profile_id
            ),
            self.settings.openai_api_key,
        ).strip()
        if self.settings.dry_run:
            return {
                "ok": False,
                "status": "dry_run",
                "detail": "Dry run mode is enabled, so no external LLM request was made.",
                "error_class": "",
                "latency_ms": None,
                "selected_model": selected_model,
                "api_base": api_base,
            }
        if not selected_model:
            return {
                "ok": False,
                "status": "not_configured",
                "detail": "Model is not configured.",
                "error_class": "not_configured",
                "latency_ms": None,
                "selected_model": "",
                "api_base": api_base,
            }
        if not api_base:
            return {
                "ok": False,
                "status": "not_configured",
                "detail": "API base is not configured.",
                "error_class": "not_configured",
                "latency_ms": None,
                "selected_model": selected_model,
                "api_base": "",
            }
        if not api_key:
            return {
                "ok": False,
                "status": "not_configured",
                "detail": "API key is not configured.",
                "error_class": "not_configured",
                "latency_ms": None,
                "selected_model": selected_model,
                "api_base": api_base,
            }

        from openai import OpenAI

        request_json: dict[str, Any] = {
            "model": selected_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a connectivity check. "
                        'Return exactly one JSON object: {"ok": true, "message": "string"}.'
                    ),
                },
                {
                    "role": "user",
                    "content": "Reply with a short confirmation message for the summary backend connectivity test.",
                },
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ConnectivityProbePayload",
                    "strict": True,
                    "schema": _ConnectivityProbePayload.model_json_schema(),
                },
            },
        }
        started = time.monotonic()
        try:
            client = OpenAI(
                base_url=_build_openai_base_url(api_base),
                api_key=api_key,
                max_retries=0,
                timeout=max(2.0, min(float(timeout_seconds or 15.0), self.settings.llm_timeout_seconds)),
            )
            response = client.chat.completions.create(**request_json)
            content = response.choices[0].message.content or "{}"
            payload = self._parse_response_payload(content=content, response_model=_ConnectivityProbePayload)
            result = _ConnectivityProbePayload.model_validate(payload)
            latency_ms = int((time.monotonic() - started) * 1000)
            return {
                "ok": bool(result.ok),
                "status": "ok" if result.ok else "error",
                "detail": result.message.strip() or "Connectivity check succeeded.",
                "error_class": "",
                "latency_ms": latency_ms,
                "selected_model": selected_model,
                "api_base": api_base,
            }
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            return {
                "ok": False,
                "status": "error",
                "detail": str(exc),
                "error_class": classify_error(exc),
                "latency_ms": latency_ms,
                "selected_model": selected_model,
                "api_base": api_base,
            }

    def _call_dry_run(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        model: str,
        response_model: type[TModel],
        response_schema: dict[str, Any] | None = None,
    ) -> TModel:
        started = time.monotonic()
        request_json = {
            "dry_run": True,
            "model": model or "dry-run",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_schema:
            request_json["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        summary_text = self._dry_summary(stage=stage, text=user)
        payload = {
            "ok": True,
            "summary": summary_text,
            "key_points": self._dry_key_points(user),
            "warnings": [],
            "source_count": max(1, user.count("\n\n") + 1),
        }
        content = json.dumps(payload, ensure_ascii=False)
        response_json = {"dry_run": True, "content": payload}
        latency_ms = int((time.monotonic() - started) * 1000)

        self.audit.write_llm_call(
            job_id=job_id,
            node_id=node_id,
            stage=stage,
            system=system,
            user=user,
            request_json=request_json,
            response_json=response_json,
            content=content,
            error=None,
            metadata={"dry_run": True, "latency_ms": latency_ms},
        )
        self.store.insert_llm_call(
            job_id=job_id,
            node_id=node_id,
            provider="dry-run",
            model=model or "dry-run",
            status="OK",
            latency_ms=latency_ms,
            prompt_tokens=estimate_tokens(system) + estimate_tokens(user),
            completion_tokens=estimate_tokens(content),
            total_tokens=estimate_tokens(system) + estimate_tokens(user) + estimate_tokens(content),
            request_json=json.dumps(request_json, ensure_ascii=False),
            response_json=json.dumps(response_json, ensure_ascii=False),
        )
        return response_model.model_validate(payload)

    def _call_openai(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        model: str,
        response_model: type[TModel],
        response_schema: dict[str, Any] | None,
        api_base: str,
        api_key: str,
        attempt: int,
    ) -> TModel:
        from openai import OpenAI

        schema = response_schema or response_model.model_json_schema()
        request_json: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        started = time.monotonic()
        response_json: dict[str, Any] | None = None
        content: str | None = None
        error: str | None = None
        pool_wait_ms = 0

        try:
            with acquire_llm_pool_slot(self.settings, job_id=job_id, node_id=node_id, stage=stage) as acquired_wait_ms:
                pool_wait_ms = acquired_wait_ms
                client = OpenAI(
                    base_url=_build_openai_base_url(api_base),
                    api_key=api_key,
                    max_retries=0,
                    timeout=self.settings.llm_timeout_seconds,
                )
                response = client.chat.completions.create(**request_json)
            if hasattr(response, "model_dump"):
                response_json = response.model_dump(mode="json")
            else:
                response_json = json.loads(response.model_dump_json())
            content = response.choices[0].message.content or "{}"
            payload = self._parse_response_payload(content=content, response_model=response_model)
            result = response_model.model_validate(payload)
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            latency_ms = int((time.monotonic() - started) * 1000)

            self.audit.write_llm_call(
                job_id=job_id,
                node_id=node_id,
                stage=stage,
                system=system,
                user=user,
                request_json=request_json,
                response_json=response_json,
                content=content,
                error=None,
                metadata={
                    "attempt": attempt,
                    "latency_ms": latency_ms,
                    "llm_pool_wait_ms": pool_wait_ms,
                    "llm_pool_max_concurrency": self.settings.llm_max_concurrency,
                },
            )
            self.store.insert_llm_call(
                job_id=job_id,
                node_id=node_id,
                provider="openai-compatible",
                model=model,
                status="OK",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                request_json=json.dumps(request_json, ensure_ascii=False),
                response_json=json.dumps(response_json, ensure_ascii=False),
            )
            return result
        except LlmPoolBusyError:
            raise
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            error = str(exc)
            latency_ms = int((time.monotonic() - started) * 1000)
            self.audit.write_llm_call(
                job_id=job_id,
                node_id=node_id,
                stage=stage,
                system=system,
                user=user,
                request_json=request_json,
                response_json=response_json,
                content=content,
                error=error,
                metadata={
                    "attempt": attempt,
                    "latency_ms": latency_ms,
                    "llm_pool_wait_ms": pool_wait_ms,
                    "llm_pool_max_concurrency": self.settings.llm_max_concurrency,
                },
            )
            self.store.insert_llm_call(
                job_id=job_id,
                node_id=node_id,
                provider="openai-compatible",
                model=model,
                status="ERROR",
                error_class=classify_error(exc),
                latency_ms=latency_ms,
                prompt_tokens=estimate_tokens(system) + estimate_tokens(user),
                request_json=json.dumps(request_json, ensure_ascii=False),
                response_json=json.dumps(response_json or {}, ensure_ascii=False),
                error_message=error,
            )
            raise

    @classmethod
    def _parse_response_payload(cls, *, content: str, response_model: type[TModel]) -> Any:
        json_text = cls._extract_json_text(content)
        if not json_text:
            raise ValueError("LLM response did not contain JSON content")
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            preview = " ".join(content.strip().split())[:240]
            raise ValueError(f"LLM response did not contain parseable JSON: {exc.msg}; preview={preview!r}") from exc
        if response_model is SummaryResult:
            return cls._normalize_summary_result_payload(payload)
        return payload

    @staticmethod
    def _extract_json_text(content: str) -> str:
        stripped = content.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped

        start_positions = [position for position in (stripped.find("{"), stripped.find("[")) if position >= 0]
        if not start_positions:
            return stripped
        start = min(start_positions)
        opening = stripped[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return stripped[start : index + 1]
        return stripped[start:]

    @classmethod
    def _normalize_summary_result_payload(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {
                "ok": True,
                "summary": cls._stringify_model_value(payload),
                "key_points": [],
                "warnings": [],
                "source_count": 1,
            }

        summary = cls._first_present(payload, "summary", "result", "answer", "content", "text")
        summary_dict = summary if isinstance(summary, dict) else {}
        key_points = payload.get("key_points", summary_dict.get("key_points"))
        warnings = payload.get("warnings", summary_dict.get("warnings"))
        source_count = payload.get("source_count", payload.get("sourceCount", summary_dict.get("source_count", 1)))
        summary_text = cls._stringify_model_value(summary)
        if not summary_text:
            fallback_payload = cls._substantive_payload(payload)
            summary_text = cls._format_mapping_summary(fallback_payload)
            if key_points is None:
                key_points = cls._key_points_from_mapping(fallback_payload)
        return {
            "ok": bool(payload.get("ok", True)),
            "summary": summary_text,
            "key_points": cls._string_list(key_points),
            "warnings": cls._string_list(warnings),
            "source_count": cls._safe_positive_int(source_count, 1),
        }

    @staticmethod
    def _first_present(payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return ""

    @staticmethod
    def _substantive_payload(payload: dict[str, Any]) -> dict[str, Any]:
        ignored = {"ok", "summary", "result", "answer", "content", "text", "key_points", "warnings", "source_count", "sourceCount"}
        return {key: value for key, value in payload.items() if key not in ignored}

    @classmethod
    def _format_mapping_summary(cls, payload: dict[str, Any]) -> str:
        lines: list[str] = []
        for key, value in payload.items():
            label = key.replace("_", " ").strip().capitalize() or "Details"
            if isinstance(value, dict):
                lines.append(f"{label}:")
                for child_key, child_value in value.items():
                    child_label = str(child_key).replace("_", " ").strip()
                    lines.append(f"- {child_label}: {cls._stringify_compact(child_value)}")
            elif isinstance(value, list):
                lines.append(f"{label}:")
                for item in value:
                    lines.append(f"- {cls._stringify_compact(item)}")
            else:
                lines.append(f"{label}: {cls._stringify_compact(value)}")
        return "\n".join(line for line in lines if line.strip())

    @classmethod
    def _key_points_from_mapping(cls, payload: dict[str, Any]) -> list[str]:
        points: list[str] = []
        for key, value in payload.items():
            label = key.replace("_", " ").strip()
            if not label:
                continue
            points.append(f"{label}: {cls._stringify_compact(value)}")
        return points

    @staticmethod
    def _stringify_compact(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return " ".join(value.split())
        if isinstance(value, (int, float, bool)):
            return str(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [cls._stringify_model_value(item) for item in value if cls._stringify_model_value(item)]
        return [cls._stringify_model_value(value)]

    @staticmethod
    def _safe_positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, parsed)

    @staticmethod
    def _stringify_model_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if isinstance(value.get("content"), str):
                return value["content"]
            if isinstance(value.get("text"), str):
                return value["text"]
        return json.dumps(value, ensure_ascii=False, indent=2)

    @staticmethod
    def _dry_summary(*, stage: str, text: str) -> str:
        compact = " ".join(text.split())
        if len(compact) > 420:
            compact = compact[:420].rstrip() + "..."
        return f"[{stage}] {compact}"

    @staticmethod
    def _dry_key_points(text: str) -> list[str]:
        lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
        return lines[:5]
