# Дизайн: MAP-REDUCE пайплайн в Dify

**Дата:** 2026-04-22  
**Статус:** черновик  

---

## 1. Цель

Перенести MAP-REDUCE пайплайн анализа логов (`log_summarizer`) в Dify Workflow (self-hosted).  
Первая версия: загрузка из ClickHouse → MAP → REDUCE → MergedAnalysis JSON.  
Отчёт, resume, multi-incident — в следующих итерациях.

---

## 2. Архитектура: 4 ноды

```
[Start] → [Code: Load & Chunk] → [Iteration: MAP] → [Code: REDUCE] → [End]
```

### 2.1 Start Node

Входные параметры workflow:

| Параметр | Тип | Описание |
|---|---|---|
| `period_start` | string | Начало периода, ISO 8601 |
| `period_end` | string | Конец периода, ISO 8601 |
| `incident_info` | string | Описание инцидента (для MAP/REDUCE промптов) |
| `alerts` | string | Тексты сработавших алертов |

---

### 2.2 Code Node: Load & Chunk

**Задача:** выгрузить логи из ClickHouse, нарезать на батчи.

**Реализация:** перенос `data_loader.py` + `chunker.py` с минимальными изменениями.

**Логика:**
1. Keyset-пагинация по `LOGS_SQLS` (два источника: containers + events) с watermark `last_ts`
2. Zone-разметка строк (`context_before` / `incident` / `context_after`)
3. Нарезка по токеновому бюджету → `list[str]`

**Конфиг (env-переменные Dify):**

| Переменная | Описание |
|---|---|
| `CH_HOST`, `CH_PORT` | ClickHouse endpoint |
| `CH_USER`, `CH_PASSWORD` | Credentials |
| `CH_DATABASE` | База данных (default: `default`) |
| `BATCH_SIZE` | Строк на страницу при пагинации (default: 200) |
| `CHUNK_TOKEN_BUDGET` | Токенов на один батч (default: 6000) |

**Выход:** `batches: list[str]`, `zones_meta: list[dict]`

---

### 2.3 Iteration Node: MAP (parallel=true)

**Задача:** параллельно проанализировать каждый батч через LLM.

**Конфигурация:**
- Тип: Iteration, параллельный режим включён
- Входной массив: `batches` из предыдущей ноды
- Дочерняя нода: LLM Node

**LLM Node внутри итерации:**
- System prompt: содержимое `log_summarizer/prompts/map_system.py` + `incident_info` + `alerts`
- User prompt: содержимое `log_summarizer/prompts/map_user.py` + текст батча
- Output mode: JSON schema (`BatchAnalysis`)
- Structured output schema: поля из `models.BatchAnalysis` (events, hypotheses, evidence, gaps)

**Выход Iteration:** `map_results: list[BatchAnalysis JSON]`

---

### 2.4 Code Node: REDUCE

**Задача:** итеративно мержить `list[BatchAnalysis]` до одного `MergedAnalysis`.

**Реализация:** перенос логики `tree_reducer.py`. LLM вызывается через `requests.post` напрямую к OpenAI-совместимому endpoint.

**Алгоритм:**
```python
items = map_results  # list[BatchAnalysis]
while len(items) > 1:
    groups = [items[i:i+group_size] for i in range(0, len(items), group_size)]
    next_items = []
    for group in groups:
        merged = call_llm_http(REDUCE_MERGE_SYSTEM, build_merge_user(group))
        next_items.append(merged)
    items = next_items
return items[0]  # MergedAnalysis
```

**Обработка ошибок:**
- Timeout: удваиваем таймаут, повторяем
- 502/503 (сервер лежит): ждём 30 с, повторяем без изменений
- Группа не влезает: дробим пополам, потом сжимаем входы
- После 5 неудачных попыток: programmatic merge (конкатенация без LLM)

**Конфиг (env-переменные Dify):**

| Переменная | Описание |
|---|---|
| `LLM_API_BASE` | Endpoint LLM (OpenAI-совместимый) |
| `LLM_API_KEY` | API ключ |
| `LLM_MODEL` | Название модели |
| `LLM_TIMEOUT` | Таймаут в секундах (default: 2400) |
| `GROUP_SIZE` | Размер группы REDUCE (default: 4) |

**Промпты:** инлайн из `log_summarizer/prompts/reduce_merge.py` и `reduce_compress.py`.

**Выход:** `merged_analysis: MergedAnalysis JSON`

---

### 2.5 End Node

Возвращает `merged_analysis` — JSON `MergedAnalysis`.

---

## 3. Зависимости Python (sandbox)

Кастомный образ `dify-sandbox`:

```dockerfile
FROM langgenius/dify-sandbox:latest
RUN pip install clickhouse-driver pydantic requests
```

В `docker-compose.yaml`:
```yaml
sandbox:
  build:
    context: .
    dockerfile: Dockerfile.sandbox
```

---

## 4. Данные между нодами

| Переход | Данные | Формат |
|---|---|---|
| Start → Load&Chunk | period_start, period_end, incident_info, alerts | строки |
| Load&Chunk → MAP | batches | `list[str]` |
| MAP → REDUCE | map_results | `list[BatchAnalysis JSON]` |
| REDUCE → End | merged_analysis | `MergedAnalysis JSON` |

---

## 5. Что НЕ входит в эту итерацию

- Генерация 14-секционного отчёта (MultipassReportGenerator)
- Resume / checkpoint при падении
- Поддержка нескольких инцидентов (CrossIncidentAnalyzer)
- UI для просмотра отчёта в Dify

---

## 6. Файлы для реализации

Переносятся/адаптируются:
- `log_summarizer/data_loader.py` → Load&Chunk Code Node
- `log_summarizer/chunker.py` → Load&Chunk Code Node  
- `log_summarizer/models.py` → инлайн в обе Code-ноды
- `log_summarizer/prompts/map_system.py`, `map_user.py` → Dify LLM Node (MAP)
- `log_summarizer/prompts/reduce_merge.py`, `reduce_compress.py` → REDUCE Code Node
- `log_summarizer/tree_reducer.py` (логика цикла) → REDUCE Code Node
- `log_summarizer/llm_client.py` (retry/backoff логика) → REDUCE Code Node
