"""Structured LLM client with JSON schema and file audit."""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .audit import AuditWriter
from .config import Settings, get_settings
from .errors import classify_error
from .logging_setup import get_logger
from .ports import AuditSink, SummaryStore
from .schemas import SummaryResult
from .text import estimate_tokens

logger = get_logger("llm_client")

TModel = TypeVar("TModel", bound=BaseModel)


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
    ) -> SummaryResult:
        return self.call_structured(
            job_id=job_id,
            node_id=node_id,
            stage=stage,
            system=system,
            user=user,
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
        response_model: type[TModel],
    ) -> TModel:
        if self.settings.dry_run:
            return self._call_dry_run(
                job_id=job_id,
                node_id=node_id,
                stage=stage,
                system=system,
                user=user,
                response_model=response_model,
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
                    response_model=response_model,
                    attempt=attempt,
                )
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

    def _call_dry_run(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        response_model: type[TModel],
    ) -> TModel:
        started = time.monotonic()
        request_json = {
            "dry_run": True,
            "model": "dry-run",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
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
            model="dry-run",
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
        response_model: type[TModel],
        attempt: int,
    ) -> TModel:
        from openai import OpenAI

        schema = response_model.model_json_schema()
        request_json: dict[str, Any] = {
            "model": self.settings.llm_model,
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

        try:
            client = OpenAI(
                base_url=_build_openai_base_url(self.settings.openai_api_base),
                api_key=self.settings.openai_api_key,
                max_retries=0,
                timeout=self.settings.llm_timeout_seconds,
            )
            response = client.chat.completions.create(**request_json)
            if hasattr(response, "model_dump"):
                response_json = response.model_dump(mode="json")
            else:
                response_json = json.loads(response.model_dump_json())
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
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
                metadata={"attempt": attempt, "latency_ms": latency_ms},
            )
            self.store.insert_llm_call(
                job_id=job_id,
                node_id=node_id,
                provider="openai-compatible",
                model=self.settings.llm_model,
                status="OK",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                request_json=json.dumps(request_json, ensure_ascii=False),
                response_json=json.dumps(response_json, ensure_ascii=False),
            )
            return result
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
                metadata={"attempt": attempt, "latency_ms": latency_ms},
            )
            self.store.insert_llm_call(
                job_id=job_id,
                node_id=node_id,
                provider="openai-compatible",
                model=self.settings.llm_model,
                status="ERROR",
                error_class=classify_error(exc),
                latency_ms=latency_ms,
                prompt_tokens=estimate_tokens(system) + estimate_tokens(user),
                request_json=json.dumps(request_json, ensure_ascii=False),
                response_json=json.dumps(response_json or {}, ensure_ascii=False),
                error_message=error,
            )
            raise

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
