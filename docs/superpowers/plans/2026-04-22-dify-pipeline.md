# Dify MAP-REDUCE Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перенести MAP-REDUCE пайплайн анализа логов в Dify Workflow (self-hosted) в виде четырёх нод: Start → Load&Chunk → MAP Iteration → REDUCE → End.

**Architecture:** ClickHouse запрашивается через HTTP-интерфейс (порт 8123) без внешних зависимостей. MAP делается нативными Dify Iteration + LLM нодами. REDUCE реализован как Python Code-нода с циклом while + HTTP-вызовами к LLM. Все Code-ноды — самодостаточные скрипты (без внешних импортов), пригодные для copy-paste в Dify.

**Tech Stack:** Python 3.11+, stdlib only (`urllib.request`, `json`, `os`), Dify self-hosted, ClickHouse HTTP interface.

---

## Структура файлов

```
dify/
  lib/
    ch_client.py      # ClickHouse HTTP client (testable unit)
    chunker.py        # Log row → chunks (testable unit)
    llm_http.py       # LLM HTTP client (testable unit)
    prompts.py        # MAP + REDUCE prompts (extracted from log_summarizer)
  nodes/
    load_chunk.py     # Dify Code Node: Load & Chunk (self-contained, copy-paste в Dify)
    reduce.py         # Dify Code Node: REDUCE (self-contained, copy-paste в Dify)
  tests/
    test_ch_client.py
    test_chunker.py
    test_llm_http.py
    test_reduce.py
  README.md           # Инструкция по настройке Dify Workflow
```

`lib/` — тестируемые модули. `nodes/` — финальные скрипты для Dify (инлайнят код из lib).

---

## Task 1: ClickHouse HTTP client

**Files:**
- Create: `dify/lib/ch_client.py`
- Create: `dify/tests/test_ch_client.py`

- [ ] **Шаг 1: Написать тест**

```python
# dify/tests/test_ch_client.py
import json
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.ch_client import ch_query

def test_ch_query_parses_jsoneachrow():
    rows = [{"timestamp": "2024-01-01", "raw_line": "foo"}]
    response_body = "\n".join(json.dumps(r) for r in rows).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = response_body

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = ch_query("localhost", 8123, "user", "pass", "SELECT 1")

    assert result == rows

def test_ch_query_skips_empty_lines():
    response_body = b'{"a": 1}\n\n{"a": 2}\n'
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = response_body

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = ch_query("localhost", 8123, "user", "pass", "SELECT 1")

    assert len(result) == 2
```

- [ ] **Шаг 2: Запустить тест — убедиться что падает**

```bash
cd /home/oleg/Documents/demo-control-plane
python -m pytest dify/tests/test_ch_client.py -v
```
Ожидаем: `ModuleNotFoundError: No module named 'lib.ch_client'`

- [ ] **Шаг 3: Реализовать**

```python
# dify/lib/ch_client.py
import json
import urllib.parse
import urllib.request


def ch_query(
    host: str,
    port: int,
    user: str,
    password: str,
    sql: str,
    timeout: int = 60,
) -> list[dict]:
    """Выполняет SQL в ClickHouse через HTTP-интерфейс, возвращает list[dict].

    ClickHouse возвращает JSONEachRow — одна строка JSON на строку ответа.
    """
    encoded = urllib.parse.quote(sql + " FORMAT JSONEachRow")
    url = f"http://{host}:{port}/?query={encoded}"
    req = urllib.request.Request(url)
    req.add_header("X-ClickHouse-User", user)
    req.add_header("X-ClickHouse-Key", password)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        lines = resp.read().decode("utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
```

- [ ] **Шаг 4: Запустить тест — убедиться что проходит**

```bash
python -m pytest dify/tests/test_ch_client.py -v
```
Ожидаем: `2 passed`

- [ ] **Шаг 5: Коммит**

```bash
git add dify/lib/ch_client.py dify/tests/test_ch_client.py
git commit -m "feat(dify): ClickHouse HTTP client"
```

---

## Task 2: Log chunker

**Files:**
- Create: `dify/lib/chunker.py`
- Create: `dify/tests/test_chunker.py`

- [ ] **Шаг 1: Написать тест**

```python
# dify/tests/test_chunker.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.chunker import chunk_rows, estimate_tokens

def test_estimate_tokens():
    # ~4 символа на токен
    assert estimate_tokens("a" * 400) == 100

def test_chunk_rows_single_chunk_when_small():
    rows = ["line1", "line2", "line3"]
    chunks = chunk_rows(rows, token_budget=1000)
    assert len(chunks) == 1
    assert "line1" in chunks[0]
    assert "line3" in chunks[0]

def test_chunk_rows_splits_when_over_budget():
    # каждая строка ~250 токенов (1000 символов), бюджет 300
    big_row = "x" * 1000
    rows = [big_row, big_row, big_row]
    chunks = chunk_rows(rows, token_budget=300)
    assert len(chunks) == 3

def test_chunk_rows_empty():
    assert chunk_rows([], token_budget=1000) == []

def test_chunk_rows_single_oversized_row():
    # строка больше бюджета — всё равно должна попасть в чанк
    big_row = "x" * 10000
    chunks = chunk_rows([big_row], token_budget=100)
    assert len(chunks) == 1
    assert big_row in chunks[0]
```

- [ ] **Шаг 2: Запустить тест — убедиться что падает**

```bash
python -m pytest dify/tests/test_chunker.py -v
```
Ожидаем: `ModuleNotFoundError: No module named 'lib.chunker'`

- [ ] **Шаг 3: Реализовать**

```python
# dify/lib/chunker.py


def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов: ~4 символа на токен."""
    return max(1, len(text) // 4)


def chunk_rows(rows: list[str], token_budget: int = 6000) -> list[str]:
    """Нарезает строки логов на батчи по токеновому бюджету.

    Каждый батч — строки соединённые '\\n', не превышающие token_budget токенов.
    Одиночная строка больше бюджета попадает в отдельный батч (не отбрасывается).
    """
    if not rows:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for row in rows:
        row_tokens = estimate_tokens(row)
        if current and current_tokens + row_tokens > token_budget:
            chunks.append("\n".join(current))
            current = []
            current_tokens = 0
        current.append(row)
        current_tokens += row_tokens

    if current:
        chunks.append("\n".join(current))

    return chunks
```

- [ ] **Шаг 4: Запустить тест — убедиться что проходит**

```bash
python -m pytest dify/tests/test_chunker.py -v
```
Ожидаем: `5 passed`

- [ ] **Шаг 5: Коммит**

```bash
git add dify/lib/chunker.py dify/tests/test_chunker.py
git commit -m "feat(dify): log chunker"
```

---

## Task 3: LLM HTTP client

**Files:**
- Create: `dify/lib/llm_http.py`
- Create: `dify/tests/test_llm_http.py`

- [ ] **Шаг 1: Написать тест**

```python
# dify/tests/test_llm_http.py
import json
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.llm_http import call_llm, LLMError

def _make_mock_response(content: str):
    body = json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = body
    return mock_resp

def test_call_llm_returns_content():
    with patch("urllib.request.urlopen", return_value=_make_mock_response('{"ok": true}')):
        result = call_llm(
            api_base="http://localhost:8000",
            api_key="test",
            model="llama",
            system="sys",
            user="user msg",
            timeout=10,
        )
    assert result == '{"ok": true}'

def test_call_llm_raises_on_http_error():
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("conn refused")):
        try:
            call_llm("http://localhost:8000", "key", "model", "sys", "user", timeout=1)
            assert False, "должно было бросить LLMError"
        except LLMError:
            pass
```

- [ ] **Шаг 2: Запустить тест — убедиться что падает**

```bash
python -m pytest dify/tests/test_llm_http.py -v
```
Ожидаем: `ModuleNotFoundError: No module named 'lib.llm_http'`

- [ ] **Шаг 3: Реализовать**

```python
# dify/lib/llm_http.py
import json
import urllib.error
import urllib.request


class LLMError(Exception):
    """LLM недоступен или вернул ошибку."""


def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout: float = 2400,
    temperature: float = 0.2,
) -> str:
    """Вызывает OpenAI-совместимый LLM API. Возвращает строку content.

    Всегда запрашивает JSON-режим (response_format: json_object).
    Raises LLMError при любой сетевой ошибке или ошибке API.
    """
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        raise LLMError(str(exc)) from exc
```

- [ ] **Шаг 4: Запустить тест — убедиться что проходит**

```bash
python -m pytest dify/tests/test_llm_http.py -v
```
Ожидаем: `2 passed`

- [ ] **Шаг 5: Коммит**

```bash
git add dify/lib/llm_http.py dify/tests/test_llm_http.py
git commit -m "feat(dify): LLM HTTP client"
```

---

## Task 4: Prompts module

**Files:**
- Create: `dify/lib/prompts.py`

Нет отдельных тестов — промпты это строки, проверяются через интеграционные тесты REDUCE.

- [ ] **Шаг 1: Создать файл с промптами**

Скопируем содержимое из `log_summarizer/prompts/` и добавим функции-билдеры:

```python
# dify/lib/prompts.py
"""MAP и REDUCE промпты для Dify-нод.

Источник: log_summarizer/prompts/{map_system,reduce_merge,reduce_compress}.py
"""

# ── MAP ───────────────────────────────────────────────────────────────

MAP_SYSTEM_TEMPLATE = """\
You are a senior SRE analyzing a production log batch during an active incident.

=== Incident context ===
{incident_context}

Period under investigation: {incident_start} → {incident_end}
{alerts_section}
=== Language ===
Think and respond in English. The incident context may be in Russian — understand it but
always output English. Keep technical terms as-is (OOM, pod names, service names,
Kubernetes objects, error codes, metric names, CLI commands).

=== Task ===
Analyze the log batch provided by the user. Extract key events, evidence, and hypotheses.
Output a single JSON object — no prose before or after.

=== Output schema ===
{{
  "time_range": ["<ISO8601>", "<ISO8601>"],
  "narrative": "<3-5 sentences: what happened in this batch>",
  "events": [
    {{
      "id": "<evt-BATCH-SEQ>",
      "timestamp": "<ISO8601>",
      "source": "<service / pod / container>",
      "severity": "<critical|high|medium|low|info>",
      "message": "<что произошло>",
      "raw_refs": ["<строка лога>"]
    }}
  ],
  "hypotheses": [
    {{
      "id": "<hyp-BATCH-SEQ>",
      "statement": "<гипотеза>",
      "confidence": "<high|medium|low>",
      "supporting_event_ids": ["<evt-id>"]
    }}
  ],
  "evidence": [
    {{"id": "<ev-BATCH-SEQ>", "text": "<факт из лога>", "event_ids": ["<evt-id>"]}}
  ],
  "gaps": ["<что не видно в этом батче>"],
  "alert_refs": []
}}"""


def build_map_system(incident_info: str, period_start: str, period_end: str, alerts: str) -> str:
    alerts_section = f"\n=== Active alerts ===\n{alerts}\n" if alerts.strip() else ""
    return MAP_SYSTEM_TEMPLATE.format(
        incident_context=incident_info,
        incident_start=period_start,
        incident_end=period_end,
        alerts_section=alerts_section,
    )


MAP_USER_TEMPLATE = """\
=== Log batch ===
{batch_text}"""


def build_map_user(batch_text: str) -> str:
    return MAP_USER_TEMPLATE.format(batch_text=batch_text)


# ── REDUCE MERGE ──────────────────────────────────────────────────────

REDUCE_MERGE_SYSTEM_TEMPLATE = """\
You are a senior SRE synthesizing partial incident analyses into a unified view.

=== Incident context ===
{incident_context}

Incident window: {incident_start} → {incident_end}

=== Language ===
Think and write all English fields in English. Keep technical terms as-is.
For every field that has a _ru counterpart, also write the Russian translation.
Russian translations must NOT translate technical terms.

=== Task ===
You are given JSON analyses of consecutive log windows. Merge them into one unified
MergedAnalysis. Deduplicate events, merge hypotheses, preserve all evidence.
Output a single JSON object — no prose before or after.

=== Output schema ===
{{
  "time_range": ["<ISO8601>", "<ISO8601>"],
  "narrative": "<unified narrative>",
  "narrative_ru": "<нарратив на русском>",
  "events": [...],
  "causal_chains": [...],
  "hypotheses": [...],
  "evidence_bank": [...],
  "gaps": [...],
  "alert_refs": [],
  "zones_covered": []
}}"""


def build_reduce_merge_system(incident_info: str, period_start: str, period_end: str) -> str:
    return REDUCE_MERGE_SYSTEM_TEMPLATE.format(
        incident_context=incident_info,
        incident_start=period_start,
        incident_end=period_end,
    )


def build_reduce_merge_user(items: list[dict]) -> str:
    import json
    parts = [f"=== Analysis {i+1} ===\n{json.dumps(item, ensure_ascii=False)}"
             for i, item in enumerate(items)]
    return "\n\n".join(parts)


# ── REDUCE COMPRESS ───────────────────────────────────────────────────

REDUCE_COMPRESS_SYSTEM = """\
You are a senior SRE compressing an incident analysis that has grown too large.

=== Language ===
Think and write all English fields in English. Keep technical terms as-is.
For every _ru field: compress its Russian text proportionally to the English compression.

=== Task ===
The MergedAnalysis JSON provided has exceeded the size budget. Compress it while
preserving all actionable signal. Output a single JSON object — no prose before or after.

Compression rules:
- narrative: shorten to 2-3 sentences, keep key facts
- events: keep only severity >= medium; truncate message to 100 chars
- hypotheses: keep top 3 by confidence
- evidence_bank: keep top 5 most relevant entries
- causal_chains: keep if they have >= 2 steps
- gaps: keep as-is (short)"""


def build_compress_user(item: dict) -> str:
    import json
    return f"=== MergedAnalysis to compress ===\n{json.dumps(item, ensure_ascii=False)}"
```

- [ ] **Шаг 2: Коммит**

```bash
git add dify/lib/prompts.py
git commit -m "feat(dify): MAP and REDUCE prompts module"
```

---

## Task 5: REDUCE logic + tests

**Files:**
- Create: `dify/lib/reduce.py`
- Create: `dify/tests/test_reduce.py`

- [ ] **Шаг 1: Написать тест**

```python
# dify/tests/test_reduce.py
import json
from unittest.mock import patch, call
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.reduce import tree_reduce, _group_items

def _make_batch_analysis(idx: int) -> dict:
    return {
        "time_range": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
        "narrative": f"batch {idx}",
        "events": [{"id": f"evt-{idx:03d}-001", "timestamp": "2024-01-01T00:00:00",
                    "source": "pod", "severity": "high", "message": "err", "raw_refs": []}],
        "hypotheses": [], "evidence": [], "gaps": [], "alert_refs": [],
    }

def _make_merged(idx: int) -> dict:
    return {
        "time_range": ["2024-01-01T00:00:00", "2024-01-01T02:00:00"],
        "narrative": f"merged {idx}", "narrative_ru": "",
        "events": [], "causal_chains": [], "hypotheses": [],
        "evidence_bank": [], "gaps": [], "alert_refs": [], "zones_covered": [],
    }

def test_group_items_splits_by_group_size():
    items = list(range(9))
    groups = _group_items(items, group_size=4)
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]

def test_tree_reduce_single_item_skips_llm():
    item = _make_batch_analysis(0)
    call_count = {"n": 0}
    def fake_call_llm(**kwargs):
        call_count["n"] += 1
        return json.dumps(_make_merged(0))

    result = tree_reduce(
        items=[item],
        call_llm_fn=fake_call_llm,
        incident_info="test", period_start="s", period_end="e",
        group_size=4,
    )
    assert call_count["n"] == 0
    assert result == item

def test_tree_reduce_two_items_calls_llm_once():
    items = [_make_batch_analysis(i) for i in range(2)]
    merged = _make_merged(0)
    call_count = {"n": 0}
    def fake_call_llm(**kwargs):
        call_count["n"] += 1
        return json.dumps(merged)

    result = tree_reduce(
        items=items,
        call_llm_fn=fake_call_llm,
        incident_info="test", period_start="s", period_end="e",
        group_size=4,
    )
    assert call_count["n"] == 1
    assert result == merged

def test_tree_reduce_eight_items_two_rounds():
    # 8 items, group_size=4 → round1: 2 LLM calls → 2 items → round2: 1 call → done
    items = [_make_batch_analysis(i) for i in range(8)]
    merged = _make_merged(0)
    call_count = {"n": 0}
    def fake_call_llm(**kwargs):
        call_count["n"] += 1
        return json.dumps(merged)

    result = tree_reduce(
        items=items,
        call_llm_fn=fake_call_llm,
        incident_info="test", period_start="s", period_end="e",
        group_size=4,
    )
    assert call_count["n"] == 3
    assert result == merged
```

- [ ] **Шаг 2: Запустить тест — убедиться что падает**

```bash
python -m pytest dify/tests/test_reduce.py -v
```
Ожидаем: `ModuleNotFoundError: No module named 'lib.reduce'`

- [ ] **Шаг 3: Реализовать**

```python
# dify/lib/reduce.py
"""Логика REDUCE-цикла для Dify Code Node.

tree_reduce() принимает list[dict] (BatchAnalysis или MergedAnalysis)
и сворачивает их в один MergedAnalysis, вызывая call_llm_fn для каждой группы.

call_llm_fn — callable(system: str, user: str) -> str (JSON-строка).
Это позволяет тестировать reduce без реального LLM.
"""
import json
import time
from typing import Callable

from prompts import build_reduce_merge_system, build_reduce_merge_user


def _group_items(items: list, group_size: int) -> list[list]:
    """Разбивает список на группы размером group_size."""
    return [items[i:i + group_size] for i in range(0, len(items), group_size)]


def _is_server_down(error_text: str) -> bool:
    t = error_text.lower()
    return "502" in t or "503" in t or "bad gateway" in t or "service unavailable" in t


def _is_timeout(error_text: str) -> bool:
    t = error_text.lower()
    return "timeout" in t or "timed out" in t


def _merge_one_group(
    group: list[dict],
    call_llm_fn: Callable,
    system: str,
    max_retries: int = 5,
    initial_timeout: float = 2400,
) -> dict:
    """Мержит одну группу через LLM с retry-логикой.

    При timeout — удваивает таймаут.
    При 502/503 — ждёт 30 с и повторяет.
    После max_retries — programmatic merge (конкатенация нарративов).
    """
    current_timeout = initial_timeout
    for attempt in range(max_retries + 1):
        try:
            raw = call_llm_fn(system=system, user=build_reduce_merge_user(group))
            return json.loads(raw)
        except Exception as exc:
            err = str(exc)
            if _is_timeout(err):
                current_timeout *= 2
            if _is_server_down(err):
                time.sleep(30)
            if attempt == max_retries:
                # Programmatic fallback: конкатенируем нарративы
                return {
                    "time_range": [
                        min(item.get("time_range", [""])[0] for item in group),
                        max(item.get("time_range", ["", ""])[1] for item in group),
                    ],
                    "narrative": " | ".join(
                        item.get("narrative", "") for item in group
                    ),
                    "narrative_ru": " | ".join(
                        item.get("narrative_ru", "") for item in group
                    ),
                    "events": [e for item in group for e in item.get("events", [])],
                    "causal_chains": [c for item in group for c in item.get("causal_chains", [])],
                    "hypotheses": [h for item in group for h in item.get("hypotheses", [])],
                    "evidence_bank": [e for item in group for e in item.get("evidence_bank", item.get("evidence", []))],
                    "gaps": list({g for item in group for g in item.get("gaps", [])}),
                    "alert_refs": [],
                    "zones_covered": [],
                }


def tree_reduce(
    items: list[dict],
    call_llm_fn: Callable,
    incident_info: str,
    period_start: str,
    period_end: str,
    group_size: int = 4,
    max_retries: int = 5,
    initial_timeout: float = 2400,
) -> dict:
    """Итеративно мержит items до одного MergedAnalysis.

    Если items уже 1 — возвращает его без LLM-вызова.
    """
    if len(items) == 1:
        return items[0]

    system = build_reduce_merge_system(incident_info, period_start, period_end)

    while len(items) > 1:
        groups = _group_items(items, group_size)
        next_items = []
        for group in groups:
            if len(group) == 1:
                next_items.append(group[0])
                continue
            merged = _merge_one_group(
                group=group,
                call_llm_fn=call_llm_fn,
                system=system,
                max_retries=max_retries,
                initial_timeout=initial_timeout,
            )
            next_items.append(merged)
        items = next_items

    return items[0]
```

- [ ] **Шаг 4: Запустить тест — убедиться что проходит**

```bash
python -m pytest dify/tests/test_reduce.py -v
```
Ожидаем: `4 passed`

- [ ] **Шаг 5: Коммит**

```bash
git add dify/lib/reduce.py dify/tests/test_reduce.py
git commit -m "feat(dify): REDUCE tree logic with retry"
```

---

## Task 6: Load & Chunk Code Node (Dify-ready)

**Files:**
- Create: `dify/nodes/load_chunk.py`

Самодостаточный скрипт для copy-paste в Dify Code Node. Содержит всё инлайн.

- [ ] **Шаг 1: Создать файл**

```python
# dify/nodes/load_chunk.py
"""Dify Code Node: Load & Chunk

Copy-paste этот файл в Dify Code Node (Python).
Входные переменные ноды: period_start, period_end, incident_info, alerts
Выходные переменные: batches (list[str]), batch_count (int)

Конфигурация через env-переменные Dify:
  CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE
  BATCH_SIZE (default: 200), CHUNK_TOKEN_BUDGET (default: 6000)
"""
import json
import os
import urllib.parse
import urllib.request


# ── ClickHouse client ────────────────────────────────────────────────

def ch_query(host, port, user, password, sql, timeout=120):
    encoded = urllib.parse.quote(sql + " FORMAT JSONEachRow")
    url = f"http://{host}:{port}/?query={encoded}"
    req = urllib.request.Request(url)
    req.add_header("X-ClickHouse-User", user)
    req.add_header("X-ClickHouse-Key", password)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        lines = resp.read().decode("utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


# ── Chunker ──────────────────────────────────────────────────────────

def estimate_tokens(text):
    return max(1, len(text) // 4)


def chunk_rows(rows, token_budget=6000):
    if not rows:
        return []
    chunks, current, current_tokens = [], [], 0
    for row in rows:
        t = estimate_tokens(row)
        if current and current_tokens + t > token_budget:
            chunks.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += t
    if current:
        chunks.append("\n".join(current))
    return chunks


# ── SQL templates ────────────────────────────────────────────────────

CONTAINERS_SQL = """
SELECT
    start_time AS timestamp,
    end_time,
    concat(
        '[', toString(start_time), ' → ', toString(end_time), ']',
        ' ×', toString(cnt),
        '  ', namespace, '/', pod_name,
        '  ', log_text
    ) AS raw_line
FROM (
    SELECT
        min(timestamp) AS start_time, max(timestamp) AS end_time,
        min(log) AS log_text, count() AS cnt,
        any(kubernetes_namespace_name) AS namespace,
        any(kubernetes_pod_name) AS pod_name,
        any(kubernetes_container_name) AS container_name
    FROM (
        SELECT *,
            sum(is_new_group) OVER (
                PARTITION BY kubernetes_namespace_name, kubernetes_container_name
                ORDER BY timestamp ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS group_id
        FROM (
            SELECT *,
                if(
                    right(log, 10) != ifNull(lagInFrame(right(log, 10)) OVER (
                        PARTITION BY kubernetes_namespace_name, kubernetes_container_name
                        ORDER BY timestamp ASC
                    ), ''), 1, 0
                ) AS is_new_group
            FROM {database}.log_k8s_containers_MT
            WHERE timestamp > parseDateTime64BestEffort('{last_ts}')
              AND timestamp <= parseDateTime64BestEffort('{period_end}')
              AND ext_ClusterName = 'ndp-p01'
              AND (
                    kubernetes_container_name LIKE '%airflow%'
                    OR kubernetes_container_name LIKE '%spark%'
                    OR kubernetes_container_name LIKE '%flex%'
                    OR (kubernetes_namespace_name LIKE '%kube-system%'
                        AND kubernetes_container_name NOT LIKE '%kube-apiserver%')
              )
              AND multiSearchAny(lower(log), [
                    'fatal','critical','error','exception','alert','panic',
                    'failed','failure','crash','abort','timeout','timed out',
                    'deadlock','out of memory','oom','disk full','no space left',
                    'permission denied','access denied','unauthorized','forbidden',
                    'connection refused','connection reset','ssl error','segfault',
                    'killed','rollback','traceback','stack trace'
              ])
            ORDER BY timestamp ASC
        )
    )
    GROUP BY group_id, kubernetes_namespace_name, kubernetes_container_name
)
ORDER BY start_time ASC
LIMIT {limit}
"""

EVENTS_SQL = """
SELECT
    timestamp,
    end_time,
    concat(
        '[EVT:', reason, ']  ',
        '[', toString(timestamp), ']',
        '  ', namespace, '  ', object_name,
        '  ', message
    ) AS raw_line
FROM (
    SELECT
        min(timestamp) AS timestamp, max(timestamp) AS end_time,
        any(reason) AS reason,
        any(kubernetes_namespace_name) AS namespace,
        any(object_name) AS object_name,
        any(message) AS message
    FROM {database}.log_k8s_events
    WHERE timestamp > parseDateTime64BestEffort('{last_ts}')
      AND timestamp <= parseDateTime64BestEffort('{period_end}')
      AND ext_ClusterName = 'ndp-p01'
      AND reason IN (
            'BackOff','ImagePullBackOff','OOMKilling','Evicted','Failed',
            'FailedCreate','FailedScheduling','FailedMount','Killing',
            'NodeNotReady','Unhealthy','CrashLoopBackOff'
      )
    GROUP BY timestamp, kubernetes_namespace_name, object_name
)
ORDER BY timestamp ASC
LIMIT {limit}
"""


# ── Main ─────────────────────────────────────────────────────────────

def main(period_start: str, period_end: str, incident_info: str, alerts: str) -> dict:
    host     = os.environ.get("CH_HOST", "localhost")
    port     = int(os.environ.get("CH_PORT", "8123"))
    user     = os.environ.get("CH_USER", "default")
    password = os.environ.get("CH_PASSWORD", "")
    database = os.environ.get("CH_DATABASE", "default")
    batch_size   = int(os.environ.get("BATCH_SIZE", "200"))
    token_budget = int(os.environ.get("CHUNK_TOKEN_BUDGET", "6000"))

    all_rows = []
    last_ts = period_start

    # Keyset pagination: загружаем постранично пока есть данные
    while True:
        page_rows = []
        for sql_tmpl in [CONTAINERS_SQL, EVENTS_SQL]:
            sql = sql_tmpl.format(
                database=database,
                last_ts=last_ts,
                period_end=period_end,
                limit=batch_size,
            )
            rows = ch_query(host, port, user, password, sql)
            page_rows.extend(rows)

        if not page_rows:
            break

        # Сортируем страницу по timestamp
        page_rows.sort(key=lambda r: r.get("timestamp", ""))
        all_rows.extend(r["raw_line"] for r in page_rows)

        # Двигаем watermark
        last_end = max(r.get("end_time", r.get("timestamp", "")) for r in page_rows)
        if last_end <= last_ts:
            break
        last_ts = last_end

        # Если страница неполная — данные закончились
        if len(page_rows) < batch_size:
            break

    batches = chunk_rows(all_rows, token_budget=token_budget)
    return {"batches": batches, "batch_count": len(batches)}
```

- [ ] **Шаг 2: Коммит**

```bash
git add dify/nodes/load_chunk.py
git commit -m "feat(dify): Load & Chunk Dify Code Node"
```

---

## Task 7: REDUCE Code Node (Dify-ready)

**Files:**
- Create: `dify/nodes/reduce.py`

- [ ] **Шаг 1: Создать файл**

```python
# dify/nodes/reduce.py
"""Dify Code Node: REDUCE

Copy-paste этот файл в Dify Code Node (Python).
Входные переменные ноды:
  map_results    — list[dict] (output Iteration ноды MAP)
  incident_info  — str
  period_start   — str
  period_end     — str
Выходные переменные: merged_analysis (dict)

Конфигурация через env-переменные Dify:
  LLM_API_BASE, LLM_API_KEY, LLM_MODEL
  LLM_TIMEOUT (default: 2400)
  GROUP_SIZE (default: 4)
"""
import json
import os
import time
import urllib.request


# ── LLM HTTP client ──────────────────────────────────────────────────

def call_llm_http(api_base, api_key, model, system, user, timeout=2400):
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


# ── Helpers ──────────────────────────────────────────────────────────

def _is_server_down(err):
    t = str(err).lower()
    return "502" in t or "503" in t or "bad gateway" in t or "service unavailable" in t

def _is_timeout(err):
    t = str(err).lower()
    return "timeout" in t or "timed out" in t

def _group_items(items, group_size):
    return [items[i:i + group_size] for i in range(0, len(items), group_size)]

def _programmatic_merge(group):
    return {
        "time_range": [
            min(item.get("time_range", [""])[0] for item in group),
            max(item.get("time_range", ["", ""])[1] for item in group),
        ],
        "narrative": " | ".join(item.get("narrative", "") for item in group),
        "narrative_ru": " | ".join(item.get("narrative_ru", "") for item in group),
        "events": [e for item in group for e in item.get("events", [])],
        "causal_chains": [c for item in group for c in item.get("causal_chains", [])],
        "hypotheses": [h for item in group for h in item.get("hypotheses", [])],
        "evidence_bank": [e for item in group for e in item.get("evidence_bank", item.get("evidence", []))],
        "gaps": list({g for item in group for g in item.get("gaps", [])}),
        "alert_refs": [],
        "zones_covered": [],
    }


# ── Prompts ──────────────────────────────────────────────────────────

def build_system(incident_info, period_start, period_end):
    return f"""\
You are a senior SRE synthesizing partial incident analyses into a unified view.

=== Incident context ===
{incident_info}

Incident window: {period_start} → {period_end}

=== Language ===
Think and write all English fields in English. Keep technical terms as-is.
For every field that has a _ru counterpart, also write the Russian translation.

=== Task ===
You are given JSON analyses of consecutive log windows. Merge them into one unified
MergedAnalysis. Deduplicate events, merge hypotheses, preserve all evidence.
Output a single JSON object — no prose before or after.

=== Output schema ===
{{
  "time_range": ["<ISO8601>", "<ISO8601>"],
  "narrative": "<unified narrative>",
  "narrative_ru": "<нарратив на русском>",
  "events": [...],
  "causal_chains": [...],
  "hypotheses": [...],
  "evidence_bank": [...],
  "gaps": [...],
  "alert_refs": [],
  "zones_covered": []
}}"""

def build_user(items):
    parts = [f"=== Analysis {i+1} ===\n{json.dumps(item, ensure_ascii=False)}"
             for i, item in enumerate(items)]
    return "\n\n".join(parts)


# ── REDUCE loop ──────────────────────────────────────────────────────

def merge_group(group, system, api_base, api_key, model, timeout, max_retries=5):
    current_timeout = timeout
    for attempt in range(max_retries + 1):
        try:
            raw = call_llm_http(api_base, api_key, model, system, build_user(group), current_timeout)
            return json.loads(raw)
        except Exception as exc:
            err = str(exc)
            if _is_timeout(err):
                current_timeout *= 2
            if _is_server_down(err):
                time.sleep(30)
            if attempt == max_retries:
                return _programmatic_merge(group)


# ── Main ─────────────────────────────────────────────────────────────

def main(map_results: list, incident_info: str, period_start: str, period_end: str) -> dict:
    api_base   = os.environ.get("LLM_API_BASE", "http://localhost:8000")
    api_key    = os.environ.get("LLM_API_KEY", "none")
    model      = os.environ.get("LLM_MODEL", "default")
    timeout    = float(os.environ.get("LLM_TIMEOUT", "2400"))
    group_size = int(os.environ.get("GROUP_SIZE", "4"))

    items = list(map_results)

    if len(items) == 1:
        return {"merged_analysis": items[0]}

    system = build_system(incident_info, period_start, period_end)

    while len(items) > 1:
        groups = _group_items(items, group_size)
        next_items = []
        for group in groups:
            if len(group) == 1:
                next_items.append(group[0])
            else:
                merged = merge_group(group, system, api_base, api_key, model, timeout)
                next_items.append(merged)
        items = next_items

    return {"merged_analysis": items[0]}
```

- [ ] **Шаг 2: Коммит**

```bash
git add dify/nodes/reduce.py
git commit -m "feat(dify): REDUCE Dify Code Node"
```

---

## Task 8: README — инструкция по настройке Dify

**Files:**
- Create: `dify/README.md`

- [ ] **Шаг 1: Создать README**

```markdown
# Dify MAP-REDUCE Pipeline

## Структура workflow

```
Start → [Code] Load & Chunk → [Iteration+LLM] MAP → [Code] REDUCE → End
```

## Настройка

### 1. Переменные среды Dify

В Dify → Settings → Environment Variables добавить:

| Переменная | Пример |
|---|---|
| CH_HOST | clickhouse.internal |
| CH_PORT | 8123 |
| CH_USER | default |
| CH_PASSWORD | secret |
| CH_DATABASE | default |
| LLM_API_BASE | http://vllm.internal:8000 |
| LLM_API_KEY | none |
| LLM_MODEL | llama-3-70b |
| LLM_TIMEOUT | 2400 |
| GROUP_SIZE | 4 |
| BATCH_SIZE | 200 |
| CHUNK_TOKEN_BUDGET | 6000 |

### 2. Start Node

Входные параметры:
- `period_start` (string) — начало периода ISO 8601
- `period_end` (string) — конец периода ISO 8601
- `incident_info` (string) — описание инцидента
- `alerts` (string) — тексты алертов

### 3. Code Node: Load & Chunk

- Тип: Code Node (Python)
- Вставить содержимое `nodes/load_chunk.py`
- Inputs: `period_start`, `period_end`, `incident_info`, `alerts`
- Outputs: `batches` (Array[String]), `batch_count` (Number)

### 4. Iteration Node: MAP

- Тип: Iteration
- Input Array: `{{load_chunk.batches}}`
- Parallel: включить
- Внутри: LLM Node
  - Model: выбрать из настроек Dify
  - System prompt: содержимое `lib/prompts.py → build_map_system(...)` (подставить переменные вручную через Jinja2)
  - User prompt: `{{item}}` (текущий батч из итерации)
  - Output format: JSON Schema (BatchAnalysis)

### 5. Code Node: REDUCE

- Тип: Code Node (Python)
- Вставить содержимое `nodes/reduce.py`
- Inputs: `map_results` (из Iteration), `incident_info`, `period_start`, `period_end`
- Outputs: `merged_analysis` (Object)

### 6. End Node

- Outputs: `merged_analysis`
```

- [ ] **Шаг 2: Коммит**

```bash
git add dify/README.md
git commit -m "docs(dify): workflow setup README"
```

---

## Task 9: Smoke test (локальный, без Dify)

Проверяем что Load&Chunk и REDUCE ноды работают как standalone Python скрипты с мокнутыми данными.

- [ ] **Шаг 1: Создать smoke test**

```python
# dify/tests/test_smoke.py
"""Smoke test: проверяем ноды как standalone функции без Dify и без реального LLM/CH."""
import json
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../nodes"))


def test_reduce_node_main_with_mock_llm():
    """REDUCE нода должна смержить 5 батчей до 1 результата."""
    import reduce as reduce_node

    merged = {
        "time_range": ["2024-01-01T00:00:00", "2024-01-01T02:00:00"],
        "narrative": "merged", "narrative_ru": "",
        "events": [], "causal_chains": [], "hypotheses": [],
        "evidence_bank": [], "gaps": [], "alert_refs": [], "zones_covered": [],
    }
    map_results = [
        {"time_range": ["2024-01-01T00:00:00", "2024-01-01T00:30:00"],
         "narrative": f"batch {i}", "events": [], "hypotheses": [],
         "evidence": [], "gaps": [], "alert_refs": []}
        for i in range(5)
    ]

    mock_resp_body = json.dumps({"choices": [{"message": {"content": json.dumps(merged)}}]}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = mock_resp_body

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch.dict(os.environ, {
            "LLM_API_BASE": "http://fake",
            "LLM_API_KEY": "test",
            "LLM_MODEL": "test",
            "GROUP_SIZE": "4",
        }):
            result = reduce_node.main(
                map_results=map_results,
                incident_info="test incident",
                period_start="2024-01-01T00:00:00",
                period_end="2024-01-01T02:00:00",
            )

    assert "merged_analysis" in result
    assert result["merged_analysis"]["narrative"] == "merged"
```

- [ ] **Шаг 2: Запустить**

```bash
python -m pytest dify/tests/test_smoke.py -v
```
Ожидаем: `1 passed`

- [ ] **Шаг 3: Финальный коммит**

```bash
git add dify/tests/test_smoke.py
git commit -m "test(dify): smoke test for REDUCE node"
```

---

## Self-Review

**Spec coverage:**
- ✅ Start node с параметрами инцидента
- ✅ Load & Chunk из ClickHouse через HTTP (без внешних deps)
- ✅ MAP через Dify Iteration (настраивается в UI, не код)
- ✅ REDUCE цикл с retry, timeout doubling, 502/503 wait, programmatic fallback
- ✅ Env-переменные для всех конфигов
- ✅ Самодостаточные Code Node скрипты для copy-paste

**Не входит в этот план (следующая итерация):**
- Генерация 14-секционного отчёта
- Resume/checkpoint
- Multi-incident
