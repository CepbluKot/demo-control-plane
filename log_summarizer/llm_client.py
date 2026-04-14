"""LLMClient — обёртка над OpenAI-совместимым API.

Единственное место, которое знает про HTTP, retry, JSON mode,
instructor и ContextOverflowError.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from log_summarizer.utils.logging import get_logger

logger = get_logger("llm_client")

# Опциональные зависимости — те же, что в my_summarizer.py
try:
    import instructor  # type: ignore[import-not-found]
except Exception:
    instructor = None  # type: ignore[assignment]

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except Exception:
    OpenAI = None  # type: ignore[assignment]

TModel = TypeVar("TModel", bound=BaseModel)


# ══════════════════════════════════════════════════════════════════════
#  Исключения
# ══════════════════════════════════════════════════════════════════════

class ContextOverflowError(Exception):
    """Контент не влез в контекст модели (HTTP 400 context_length_exceeded).

    Сигнал для вызывающего кода: нужен split.
    """


class LLMUnavailableError(Exception):
    """LLM недоступна после всех retry (500/timeout)."""


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _is_context_overflow(exc: Exception) -> bool:
    """True если это ошибка переполнения контекста."""
    status = getattr(exc, "status_code", None)
    if status == 400:
        text = str(exc).lower()
        # vLLM / OpenAI формулировки
        for marker in (
            "context_length_exceeded",
            "context length",
            "maximum context",
            "prompt is too long",
            "input is too long",
            "invalid grammar",   # vLLM grammar overflow
        ):
            if marker in text:
                return True
    return False


def _is_retryable(exc: Exception) -> bool:
    """True если ошибку стоит retry-ть (500/502/503/504/timeout)."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in (500, 502, 503, 504):
        return True
    text = str(exc).lower()
    for marker in ("timeout", "read timed out", "connection error", "gateway"):
        if marker in text:
            return True
    return False


def _build_openai_base_url(api_base: str) -> str:
    """Убеждаемся что base_url заканчивается на /v1."""
    url = str(api_base or "").rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


# ══════════════════════════════════════════════════════════════════════
#  LLMClient
# ══════════════════════════════════════════════════════════════════════

class LLMClient:
    """Обёртка над OpenAI-совместимым API.

    Поддерживает:
    - JSON mode через instructor или прямой разбор JSON
    - Plain text mode для финального отчёта
    - Retry с exponential backoff на 500/timeout
    - ContextOverflowError на 400 context_length_exceeded
    - Автоматический fallback с TOOLS → JSON при грамматических ошибках vLLM
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
        use_instructor: bool = True,
        model_supports_tool_calling: bool = False,
        timeout: float = 600.0,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.use_instructor = use_instructor
        self.model_supports_tool_calling = model_supports_tool_calling
        self.timeout = timeout

        # Кэш инструктор-клиентов (TOOLS / JSON)
        self._instructor_cache: dict[str, Any] = {}
        # Если TOOLS даёт grammar error — переходим в JSON mode навсегда
        self._force_json_mode: bool = False

    # ── Публичный API ─────────────────────────────────────────────────

    async def call_json(
        self,
        system: str,
        user: str,
        response_model: Type[TModel],
        temperature: float = 0.2,
    ) -> TModel:
        """Вызов с JSON-выводом, парсинг через Pydantic.

        Raises:
            ContextOverflowError: Контент не влез в контекст.
            LLMUnavailableError: Не удалось получить ответ после retry.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._call_json_sync(system, user, response_model, temperature),
        )

    async def call_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
    ) -> str:
        """Вызов с plain text ответом (для финального отчёта).

        Raises:
            ContextOverflowError: Контент не влез в контекст.
            LLMUnavailableError: Не удалось получить ответ после retry.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._call_text_sync(system, user, temperature),
        )

    # ── Синхронные реализации (executor) ─────────────────────────────

    def _call_json_sync(
        self,
        system: str,
        user: str,
        response_model: Type[TModel],
        temperature: float,
    ) -> TModel:
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):
            try:
                return self._do_call_json(system, user, response_model, temperature)
            except ContextOverflowError:
                raise  # не retry-ть, сразу пробросить
            except Exception as exc:
                last_exc = exc
                if _is_context_overflow(exc):
                    raise ContextOverflowError(str(exc)) from exc
                if not _is_retryable(exc) and attempt == 1:
                    # Одна попытка для нестандартных ошибок JSON-parse
                    if "json" in str(exc).lower() or "parse" in str(exc).lower():
                        logger.warning(
                            "JSON parse error (attempt %d), retrying with temperature=0: %s",
                            attempt, exc,
                        )
                        try:
                            return self._do_call_json(system, user, response_model, 0.0)
                        except Exception as exc2:
                            if _is_context_overflow(exc2):
                                raise ContextOverflowError(str(exc2)) from exc2
                            last_exc = exc2
                    break
                if attempt <= self.max_retries:
                    delay = self.retry_backoff_base ** attempt
                    logger.warning(
                        "LLM call_json error (attempt %d/%d), retry in %.1fs: %s",
                        attempt, self.max_retries + 1, delay, exc,
                    )
                    time.sleep(delay)

        raise LLMUnavailableError(
            f"LLM call_json failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    def _call_text_sync(
        self,
        system: str,
        user: str,
        temperature: float,
    ) -> str:
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):
            try:
                return self._do_call_text(system, user, temperature)
            except ContextOverflowError:
                raise
            except Exception as exc:
                last_exc = exc
                if _is_context_overflow(exc):
                    raise ContextOverflowError(str(exc)) from exc
                if attempt <= self.max_retries:
                    delay = self.retry_backoff_base ** attempt
                    logger.warning(
                        "LLM call_text error (attempt %d/%d), retry in %.1fs: %s",
                        attempt, self.max_retries + 1, delay, exc,
                    )
                    time.sleep(delay)

        raise LLMUnavailableError(
            f"LLM call_text failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    # ── Внутренние вызовы ─────────────────────────────────────────────

    def _do_call_json(
        self,
        system: str,
        user: str,
        response_model: Type[TModel],
        temperature: float,
    ) -> TModel:
        """Один LLM-вызов с JSON-выводом."""
        if self.use_instructor and instructor is not None and OpenAI is not None:
            return self._call_via_instructor(system, user, response_model, temperature)
        return self._call_json_direct(system, user, response_model, temperature)

    def _do_call_text(
        self,
        system: str,
        user: str,
        temperature: float,
    ) -> str:
        """Один LLM-вызов с plain text выводом."""
        if OpenAI is None:
            raise LLMUnavailableError("openai package not installed")
        client = OpenAI(
            base_url=_build_openai_base_url(self.api_base),
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def _call_via_instructor(
        self,
        system: str,
        user: str,
        response_model: Type[TModel],
        temperature: float,
    ) -> TModel:
        """Вызов через instructor с fallback TOOLS→JSON при grammar ошибках."""
        client = self._get_instructor_client()
        try:
            result = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_model=response_model,
                temperature=temperature,
                timeout=self.timeout,
            )
            return result
        except Exception as exc:
            # vLLM grammar error → переключиться в JSON mode и retry
            exc_text = str(exc).lower()
            if (
                not self._force_json_mode
                and (
                    "tool_choice" in exc_text
                    and "tool-call-parser" in exc_text
                    or "invalid grammar" in exc_text
                )
            ):
                logger.warning(
                    "Grammar/tool-call parser error, switching to JSON mode: %s", exc
                )
                self._force_json_mode = True
                # Инвалидируем кэш и повторяем с JSON mode
                self._instructor_cache.clear()
                client = self._get_instructor_client()
                return client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_model=response_model,
                    temperature=temperature,
                    timeout=self.timeout,
                )
            raise

    def _call_json_direct(
        self,
        system: str,
        user: str,
        response_model: Type[TModel],
        temperature: float,
    ) -> TModel:
        """Прямой вызов с response_format=json_object и ручным парсингом."""
        if OpenAI is None:
            raise LLMUnavailableError("openai package not installed")
        client = OpenAI(
            base_url=_build_openai_base_url(self.api_base),
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return response_model.model_validate_json(content)

    def _get_instructor_client(self) -> Any:
        """Возвращает (или создаёт) instructor-клиент с нужным mode."""
        if instructor is None or OpenAI is None:
            raise LLMUnavailableError("instructor or openai package not installed")

        mode = self._resolve_mode()
        cache_key = f"{self.api_base}|{self.model}|{mode}"
        if cache_key in self._instructor_cache:
            return self._instructor_cache[cache_key]

        openai_client = OpenAI(
            base_url=_build_openai_base_url(self.api_base),
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )
        mode_enum = getattr(instructor, "Mode", None)
        if mode == "TOOLS" and mode_enum and hasattr(mode_enum, "TOOLS"):
            client = instructor.from_openai(openai_client, mode=mode_enum.TOOLS)
        elif mode_enum and hasattr(mode_enum, "JSON"):
            client = instructor.from_openai(openai_client, mode=mode_enum.JSON)
        else:
            client = instructor.from_openai(openai_client)

        self._instructor_cache[cache_key] = client
        return client

    def _resolve_mode(self) -> str:
        """Определяем режим instructor: TOOLS или JSON."""
        if self._force_json_mode:
            return "JSON"
        if self.model_supports_tool_calling:
            return "TOOLS"
        return "JSON"
