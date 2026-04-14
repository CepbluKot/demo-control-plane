#!/usr/bin/env python3
"""
Мини-тест: проверяем, умеет ли текущая модель/ручка работать с tool-calling через instructor.

Запуск:
    python debug_instructor_tool_calling.py

Никаких CLI-аргументов — все настройки в константах ниже.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from typing import Any, List

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator


# ===== НАСТРОЙКИ (заполни под себя) =====
API_BASE = "https://phoenix.scm-test.int.gazprombank.ru/api/v1"
API_KEY = "PUT_YOUR_KEY_HERE"
MODEL_ID = "PNX.QWEN3 235b a22b instruct"
TIMEOUT_SEC = 120.0
TEMPERATURE = 0.0


class ToolCheckEvent(BaseModel):
    id: str = Field(description="ID события, например evt-001")
    timestamp: str = Field(description="ISO8601, например 2026-04-14T10:00:00Z")
    source: str
    severity: str
    importance: float = Field(ge=0.0, le=1.0)
    description: str

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        text = str(value or "").strip()
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timestamp must include timezone (Z or ±HH:MM)")
        return text


class ToolCheckPayload(BaseModel):
    ok: bool
    summary: str
    timeline: List[ToolCheckEvent]


def _build_openai_client() -> OpenAI:
    return OpenAI(base_url=API_BASE.rstrip("/"), api_key=API_KEY, max_retries=0)


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _safe_schema() -> dict[str, Any]:
    schema = ToolCheckPayload.model_json_schema()
    # В tool schema не нужен title/defs от pydantic — убираем лишнее, чтобы гейтвей не спотыкался.
    schema.pop("title", None)
    return schema


def test_raw_tools_call() -> bool:
    """Проверка: endpoint/model вообще поддерживают tools (без instructor)."""
    _print_header("1) RAW OpenAI tools check")
    client = _build_openai_client()
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            temperature=TEMPERATURE,
            timeout=TIMEOUT_SEC,
            messages=[
                {
                    "role": "system",
                    "content": "Ты возвращаешь данные только через вызов функции emit_payload.",
                },
                {
                    "role": "user",
                    "content": (
                        "Сформируй 2 события в timeline. "
                        "timestamp строго валидный ISO8601. "
                        "importance в диапазоне 0..1."
                    ),
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "emit_payload",
                        "description": "Вернуть результат проверки в структуре ToolCheckPayload",
                        "parameters": _safe_schema(),
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "emit_payload"}},
        )
        elapsed = time.perf_counter() - started
        msg = response.choices[0].message if response.choices else None
        tool_calls = getattr(msg, "tool_calls", None) if msg else None
        if not tool_calls:
            print(f"FAIL: tools не вернулись (elapsed={elapsed:.2f}s)")
            print("Assistant content:", getattr(msg, "content", None) if msg else None)
            return False
        print(f"OK: tools поддерживаются (elapsed={elapsed:.2f}s, tool_calls={len(tool_calls)})")
        args_raw = tool_calls[0].function.arguments
        print("Tool args preview:", str(args_raw)[:500])
        return True
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        print(f"FAIL: RAW tools check упал (elapsed={elapsed:.2f}s)")
        print(f"Error: {type(exc).__name__}: {exc}")
        return False


def test_instructor_mode(mode_name: str, mode_value: Any) -> bool:
    _print_header(f"2) Instructor mode check: {mode_name}")
    started = time.perf_counter()
    try:
        openai_client = _build_openai_client()
        if mode_value is not None:
            client = instructor.from_openai(openai_client, mode=mode_value)
        else:
            client = instructor.from_openai(openai_client)

        result: ToolCheckPayload = client.chat.completions.create(
            model=MODEL_ID,
            response_model=ToolCheckPayload,
            temperature=TEMPERATURE,
            timeout=TIMEOUT_SEC,
            max_retries=0,
            messages=[
                {
                    "role": "system",
                    "content": "Верни строго структуру ToolCheckPayload.",
                },
                {
                    "role": "user",
                    "content": (
                        "Сформируй валидный payload с двумя событиями. "
                        "Используй только валидные ISO8601 timestamp."
                    ),
                },
            ],
        )

        elapsed = time.perf_counter() - started
        print(f"OK: Instructor {mode_name} сработал (elapsed={elapsed:.2f}s)")
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)[:1200])
        return True
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        print(f"FAIL: Instructor {mode_name} упал (elapsed={elapsed:.2f}s)")
        print(f"Error: {type(exc).__name__}: {exc}")
        text = str(exc).lower()
        if "tool_choice" in text or "tool-call-parser" in text or "invalid grammar" in text:
            print("Hint: похоже, gateway/model не поддерживают TOOLS mode стабильно.")
        if "choices" in text and "none" in text:
            print("Hint: похоже, backend вернул malformed payload (choices=None).")
        print("Traceback:")
        traceback.print_exc(limit=2)
        return False


def main() -> int:
    if API_KEY.strip() in {"", "PUT_YOUR_KEY_HERE"}:
        print("Заполни API_KEY в debug_instructor_tool_calling.py")
        return 2

    print("Model:", MODEL_ID)
    print("Base:", API_BASE)

    raw_ok = test_raw_tools_call()

    mode_tools = getattr(getattr(instructor, "Mode", object()), "TOOLS", None)
    mode_json = getattr(getattr(instructor, "Mode", object()), "JSON", None)

    tools_ok = False
    json_ok = False

    if mode_tools is not None:
        tools_ok = test_instructor_mode("TOOLS", mode_tools)
    else:
        print("Instructor.Mode.TOOLS не найден в текущей версии instructor")

    if mode_json is not None:
        json_ok = test_instructor_mode("JSON", mode_json)
    else:
        print("Instructor.Mode.JSON не найден в текущей версии instructor")

    _print_header("RESULT")
    print(f"RAW tools:        {'OK' if raw_ok else 'FAIL'}")
    print(f"Instructor TOOLS: {'OK' if tools_ok else 'FAIL'}")
    print(f"Instructor JSON:  {'OK' if json_ok else 'FAIL'}")

    if raw_ok and tools_ok:
        print("Итог: tool-calling через instructor работает.")
        return 0

    if raw_ok and not tools_ok and json_ok:
        print("Итог: endpoint tools поддерживает, но Instructor TOOLS нестабилен для этой модели/гейтвея.")
        print("Рекомендация: использовать Instructor JSON mode в проде.")
        return 1

    print("Итог: с tool-calling есть проблемы на стороне модели/гейтвея или конфигурации.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
