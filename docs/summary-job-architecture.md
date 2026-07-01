# Архитектура summary job pipeline

**Статус:** целевая архитектура MVP  
**Контур:** закрытая сеть, executor запускается рядом с control plane  
**Основная цель:** суммаризация большого контекста, который не помещается в один LLM-вызов, с наблюдаемым pipeline, checkpoint/resume и сохранением промежуточных результатов.

---

## 1. Итоговый стек

| Слой | Выбор для MVP | Зачем |
|---|---|---|
| API / control plane | FastAPI + Pydantic | Создание задач, управление pause/resume/cancel, чтение статусов и артефактов |
| Frontend | FastAPI static service + vanilla JS | Отдельный сервис UI без npm-сборки, REST + WebSocket к backend |
| Executor | Dramatiq workers | Готовая очередь задач, retry, worker pool, без самописного executor loop |
| Broker | Redis | Доставка фоновых задач между API и workers |
| State / artifacts | ClickHouse 24.8 | Event log, промежуточные summaries, финальные результаты, audit LLM calls |
| LLM client | OpenAI-compatible HTTP/API client | Прямые вызовы без LangChain/LlamaIndex/Instructor в новом модуле |
| Structured output | Provider-native `response_format=json_schema` | Схемная генерация JSON на стороне LLM gateway |
| Validation | `json.loads` + Pydantic `model_validate` | Локальная страховка, понятные ошибки и retry |
| Chunking | char-based estimate, без tokenizer | Быстро, дёшево, без локальных model/tokenizer assets |

---

## 2. Что не используем в MVP

### Airflow

Пока не используем как executor, потому что executor должен работать в том же контуре и с теми же сетевыми доступами, что и control plane. У Airflow в целевом окружении может не быть доступа до LLM endpoint.

Архитектурно оставляем возможность позже добавить `AirflowExecutor`, но MVP строим на Dramatiq.

### LangChain

Не используем как основной слой, потому что задача не agent/RAG/tool workflow. Главные требования здесь: checkpoint/resume, idempotency, прозрачные артефакты, pause/resume и контроль retry. Эти вещи всё равно пришлось бы реализовывать самим.

### LlamaIndex

Не используем `tree_summarize`, потому что не хотим тянуть большой framework ради простой MAP/REDUCE-схемы.

### Instructor

Не используем в новом модуле как основной путь. Старый код, где Instructor уже есть, можно оставить. Для новой части основной путь:

```text
LLM response_format=json_schema
-> message.content
-> json.loads
-> Pydantic validation
```

### Tokenizer

Не используем tokenizer/token counter. Оценка размера делается по символам с большим safety margin. Если LLM всё равно вернула context overflow, chunk делится и повторяется.

---

## 3. High-level схема

```text
User / UI
  |
  v
Frontend service
  |
  | REST + WebSocket
  v
FastAPI control plane
  |
  | writes job/input/events
  v
ClickHouse 24.8  <-----------------------------+
  ^                                             |
  | reads state/artifacts                       |
  | writes events/artifacts/llm calls           |
  |                                             |
Dramatiq workers  <---- Redis broker -----------+
  |
  v
OpenAI-compatible LLM endpoint
```

Control plane не выполняет долгую работу внутри HTTP request. Он только создаёт job, пишет события, ставит задачи в очередь и отдаёт состояние пользователю.

Dramatiq workers выполняют pipeline рядом с control plane, поэтому имеют тот же сетевой доступ до LLM.

ClickHouse является источником правды по pipeline state. Broker не является источником правды.

---

## 4. Основной pipeline

```text
1. User creates job
2. Backend writes JOB_CREATED and input artifact to ClickHouse
3. Backend enqueues advance_job(job_id)
4. advance_job creates chunks and MAP nodes
5. map_node summarizes each chunk
6. advance_job creates REDUCE level 1
7. reduce_node merges groups of summaries
8. advance_job creates REDUCE level N+1 while more than one summary remains
9. finalize_job creates final summary
10. Backend shows pipeline progress and artifacts from ClickHouse
```

Reduce уровней может быть сколько угодно:

```text
237 MAP summaries
-> REDUCE level 1: 30 summaries
-> REDUCE level 2: 4 summaries
-> REDUCE level 3: 1 summary
-> FINAL
```

### 4.1. Стратегия JSON-схем

Pipeline поддерживает два независимых структурированных режима:

| Уровень | Metadata key | Как работает |
|---|---|---|
| Промежуточный `MAP` + `REDUCE` | `intermediate_output_json_schema` | Одна shared JSON Schema для обоих этапов. `MAP` возвращает объект этой схемы для одного чанка, `REDUCE` возвращает объект той же схемы для группы partial summaries |
| Финальный `FINAL` | `output_json_schema` | Пользовательская JSON Schema только для финального ответа |

Почему для `MAP` и `REDUCE` используется одна shared schema:

- не требуется отдельная логика преобразования `map -> reduce`;
- проще гарантировать merge совместимых объектов;
- проще валидировать артефакты и повторять шаги;
- удобнее анализировать sequence-sensitive данные вроде логов и таймсерий.

Если `intermediate_output_json_schema` не задана, `MAP` и `REDUCE` продолжают использовать внутренний transport shape:

```json
{"ok": true, "summary": "string", "key_points": ["string"], "warnings": ["string"], "source_count": 1}
```

Если `intermediate_output_json_schema` задана:

- `MAP` и `REDUCE` вызываются через structured output с provider-native `json_schema`;
- user prompt templates могут использовать placeholder `{intermediate_output_json_schema}`;
- промежуточные артефакты хранят сам JSON-объект и metadata о sequence диапазоне;
- порядок чанков сохраняется и для `REDUCE` grouping, и для последующих reduce-level merges.

---

## 5. Dramatiq actors

Минимальный набор actors:

```python
@dramatiq.actor
def advance_job(job_id: str) -> None:
    ...

@dramatiq.actor
def map_node(job_id: str, node_id: str) -> None:
    ...

@dramatiq.actor
def reduce_node(job_id: str, node_id: str) -> None:
    ...

@dramatiq.actor
def finalize_job(job_id: str) -> None:
    ...

@dramatiq.actor
def recover_jobs() -> None:
    ...
```

Назначение:

| Actor | Ответственность |
|---|---|
| `advance_job` | Координатор. Смотрит состояние job в ClickHouse и ставит следующие runnable tasks |
| `map_node` | Обрабатывает один chunk или batch chunks |
| `reduce_node` | Объединяет группу MAP/REDUCE summaries |
| `finalize_job` | Создаёт финальный ответ |
| `recover_jobs` | После рестарта переотправляет незавершённые runnable jobs |

Dramatiq хранит только техническое состояние доставки задач: queued, delayed retry, failed/dead-letter. Бизнес-состояние pipeline хранится в ClickHouse.

---

## 6. State model

### Job states

```text
CREATED
RUNNING
PAUSE_REQUESTED
PAUSED
RESUMED
CANCEL_REQUESTED
CANCELLED
WAITING_RETRY
WAITING_PROVIDER
FAILED
DONE
```

### Node states

```text
PENDING
QUEUED
RUNNING
PAUSED
WAITING_RETRY
DONE
FAILED_RETRYABLE
FAILED_FINAL
SKIPPED_ALREADY_DONE
```

### Node types

```text
CHUNK
MAP
REDUCE
FINAL
```

---

## 7. ClickHouse storage model

ClickHouse 24.8 не используем как mutable OLTP state store. Состояние пишется append-only событиями. Текущий статус считается как последнее событие по `job_id` / `node_id`.

Не используем новый `JSON` type как базовую опору. Для 24.8 безопаснее:

```text
important fields -> typed columns
payload/result/error/raw_request/raw_response -> String with JSON
```

Рекомендуемые таблицы:

| Таблица | Что хранит |
|---|---|
| `summary_job_events` | События job: created, pause, resume, failed, done |
| `summary_node_events` | События node: pending, started, done, failed, retry |
| `summary_artifacts` | Input, chunks, MAP summaries, REDUCE summaries, final summary |
| `summary_input_segments` | Нормализованные сегменты входных логов из upload/query до создания MAP |
| `summary_llm_calls` | Audit каждого LLM-вызова: provider, model, latency, usage, error |

Принцип: после каждого успешного LLM-вызова artifact сразу записывается в ClickHouse. Ничего важного не живёт только в памяти worker-а.

---

## 8. Idempotency и resume

Каждый node получает детерминированный id:

```text
node_id = hash(job_id + node_type + level + index + input_hash)
```

Перед выполнением actor проверяет ClickHouse:

```text
if successful artifact for node_id exists:
    write SKIPPED_ALREADY_DONE
    return
```

Это защищает от:

- повторной доставки Dramatiq task;
- падения worker-а после LLM-вызова, но до broker ack;
- рестарта backend/worker;
- ручного recovery;
- повторного запуска `advance_job`.

После рестарта worker запускает recovery:

```text
find jobs in RUNNING / WAITING_RETRY / WAITING_PROVIDER
-> enqueue advance_job(job_id)
```

Job в `PAUSED` не продолжается автоматически, пока пользователь не отправит resume.

---

## 9. Pause, resume, cancel

Pause:

```text
POST /jobs/{job_id}/pause
-> write JOB_PAUSE_REQUESTED
```

Workers проверяют job state:

- перед каждым LLM call;
- между retry;
- перед созданием следующего reduce level;
- перед `finalize_job`.

Если pause requested:

```text
write NODE_PAUSED / JOB_PAUSED
stop current actor without new LLM call
```

Resume:

```text
POST /jobs/{job_id}/resume
-> write JOB_RESUMED
-> enqueue advance_job(job_id)
```

Cancel:

```text
POST /jobs/{job_id}/cancel
-> write JOB_CANCEL_REQUESTED
-> workers stop scheduling new work
-> write JOB_CANCELLED when no active node should continue
```

---

## 10. LLM reliability policy

Ошибки классифицируются явно:

| Ошибка | Действие |
|---|---|
| Rate limit / 429 | Retry with backoff, учитывать `Retry-After` если есть |
| Provider down / 5xx / timeout | Long backoff, circuit breaker, job может перейти в `WAITING_PROVIDER` |
| Quota exhausted | `WAITING_PROVIDER` или `FAILED_FINAL`, зависит от политики |
| Context too long | Split chunk/group smaller and retry |
| Invalid JSON | Retry with validation error in prompt |
| Pydantic schema error | Retry, затем `FAILED_RETRYABLE` / `FAILED_FINAL` |
| Fatal bad request | Fail node/job без бесконечного retry |

Для нестабильного LLM endpoint нужен circuit breaker:

```text
if N provider failures in short window:
    stop scheduling new LLM calls
    mark jobs WAITING_PROVIDER
    retry advance_job later
```

Concurrency ограничивается на уровне Dramatiq workers/queues и отдельным Redis-backed global limiter для LLM. Лимитер общий для всех jobs, worker threads, worker processes и pod-ов, которые используют один `SUMMARY_BACKEND_BROKER_URL`; настройка `SUMMARY_BACKEND_LLM_MAX_CONCURRENCY` задает максимум одновременных outbound LLM HTTP calls.

---

## 11. Structured output contract

Все LLM-вызовы в новом модуле используют provider-native schema:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "summary_result",
      "strict": true,
      "schema": {
        "type": "object",
        "additionalProperties": false
      }
    }
  }
}
```

После ответа:

```python
content = response["choices"][0]["message"]["content"]
payload = json.loads(content)
result = SummaryModel.model_validate(payload)
```

Локальная Pydantic-валидация обязательна даже при рабочем `json_schema`, потому что gateway/model могут деградировать или вернуть нестандартный ответ.

---

## 12. Chunking policy

Tokenizer не используется.

Базовая оценка:

```python
estimated_tokens = ceil(len(text) / 2.2)
```

Сборка chunks:

```text
1. Split input into atomic units: paragraphs, lines, log rows.
2. Estimate each atomic unit once.
3. Build chunks by summing cached estimates.
4. Keep safety margin below model context limit.
5. On context overflow, split chunk/group and retry.
```

Размеры должны быть консервативными. Лучше сделать больше MAP-вызовов, чем регулярно ловить overflow.

---

## 13. API surface MVP

Минимальные endpoints:

```text
POST   /summary-jobs
POST   /summary-jobs/upload
POST   /summary-jobs/clickhouse-query
GET    /summary-jobs/{job_id}
GET    /summary-jobs/{job_id}/events
GET    /summary-jobs/{job_id}/nodes
GET    /summary-jobs/{job_id}/artifacts
GET    /summary-jobs/{job_id}/input-segments
POST   /summary-jobs/{job_id}/pause
POST   /summary-jobs/{job_id}/resume
POST   /summary-jobs/{job_id}/cancel
```

UI должен строиться только по данным из ClickHouse:

```text
job timeline
stage progress
MAP node outputs
REDUCE level outputs
LLM call errors/retries
final summary
```

---

## 14. Главный инвариант

```text
Dramatiq = доставить работу
ClickHouse = источник правды
LLM result = artifact, сохранённый сразу после node
```

Если этот инвариант соблюдается, pipeline можно безопасно продолжать после падения backend, worker, broker-delivery duplicate или временной недоступности LLM.

---

## 15. Модульные границы

Основная state machine не должна зависеть напрямую от ClickHouse, Dramatiq или конкретного LLM SDK. Для этого backend использует ports/protocols:

| Port | MVP adapter | Замена без переписывания pipeline |
|---|---|---|
| `SummaryStore` | ClickHouse append-only store | Postgres, S3+DB, другая event/artifact storage |
| `TaskQueue` | Dramatiq/Redis | Celery, Airflow adapter, inline executor |
| `SummaryLLM` | OpenAI-compatible `json_schema` client | другой gateway, mock, batch client |
| `Chunker` | char-budget chunker | tokenizer-based splitter, semantic splitter |
| `InputSegmenter` | row-budget input segmenter | tokenizer-aware segmenter, semantic row grouper |
| `AuditSink` | file audit writer | object storage audit, no-op audit, encrypted audit |

Файловый ввод находится перед state machine:

```text
Upload CSV / Markdown / JSON
-> staging file artifacts/summary_backend/uploads/{job_id}/...
-> ingest_upload Dramatiq actor
-> UploadedLogParser
-> normalized LogRecord stream
-> InputSegmenter
-> summary_input_segments
-> INPUT_READY
-> advance_job creates MAP nodes from persisted segments
```

Текстовый ввод и ClickHouse query идут туда же:

```text
POST /summary-jobs input_text
-> LogRecord units
-> InputSegmenter
-> summary_input_segments
-> MAP nodes

POST /summary-jobs/clickhouse-query
-> ClickHouse query_row_block_stream
-> normalize rows to LogRecord
-> InputSegmenter
-> summary_input_segments
-> MAP nodes
```

Fallback chunking из `summary_artifacts.INPUT.content` оставлен только для legacy jobs, созданных старым кодом.

Default wiring находится отдельно от core pipeline. Это позволяет тестировать pipeline на fake/in-memory реализациях и менять инфраструктурные адаптеры без изменения бизнес-логики MAP/REDUCE.
