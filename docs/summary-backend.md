# Summary backend MVP

Этот backend реализует самостоятельный job pipeline для больших входных логов.

## Состав

```text
summary_backend/
  api.py              FastAPI control plane
  snapshots.py        UI/WebSocket read model
  tasks.py            Dramatiq actors
  ports.py            Protocols/interfaces for replaceable components
  factory.py          Default wiring of concrete adapters
  pipeline.py         MAP/REDUCE state machine
  ingestion.py        Create jobs from uploaded files
  input_parsers.py    CSV / Markdown / JSON log parsers
  query_sources.py    Streaming ClickHouse query readers
  input_segments.py   Durable input segment builder
  input_models.py     Normalized log row and input segment models
  store.py            ClickHouse event/artifact store
  llm_client.py       OpenAI-compatible json_schema calls
  audit.py            Per-call files: prompts, request, response, errors
  logging_setup.py    Console + rotating file logs

summary_frontend/
  app.py              FastAPI service for static frontend files
  static/             HTML/CSS/JS dashboard
```

## Модульность

Core pipeline зависит от protocol-интерфейсов из `summary_backend/ports.py`:

| Port | Текущая реализация | Что можно заменить |
|---|---|---|
| `SummaryStore` | `ClickHouseStore` | Postgres, S3+DB, другая event storage схема |
| `TaskQueue` | `DramatiqTaskQueue` | Celery, Airflow adapter, inline/local executor |
| `SummaryLLM` | `StructuredLLMClient` | другой LLM gateway, mock, batch client |
| `Chunker` | `CharBudgetChunker` | tokenizer-based chunker, semantic splitter |
| `InputSegmenter` | `RowBudgetInputSegmenter` | другой row grouper, tokenizer-aware segmenter |
| `AuditSink` | `AuditWriter` | file audit, object storage audit, no-op audit |

Default wiring находится в `summary_backend/factory.py`. Для тестов или новой инфраструктуры можно передать свои реализации в `PipelineService` или `create_pipeline_service(...)`.

Парсинг файлов вынесен отдельно от pipeline:

| Абстракция | Текущая реализация |
|---|---|
| `UploadedLogParser` | `CsvLogParser`, `MarkdownTableLogParser`, `JsonLogParser` |
| `ParserRegistry` | Определяет формат по extension/content-type/form field |
| `UploadedFileIngestionService` | Создаёт job, пишет input segments, ставит job в очередь |
| `StagedUploadIngestionService` | Сохраняет upload в staging, фоновой задачей пишет input segments |
| `QueryLogRecordSource` | `ClickHouseQueryLogRecordSource` читает rows блоками |
| `ClickHouseQueryIngestionService` | Создаёт job из SQL result, пишет input segments |

## Runtime model

```text
FastAPI
  POST /summary-jobs
  -> splits input_text into LogRecord units
  -> writes JOB_CREATED + summary_input_segments + INPUT manifest
  -> enqueues advance_job(job_id)

FastAPI
  POST /summary-jobs/clickhouse-query
  -> streams ClickHouse query rows
  -> normalizes rows with the same LogRecord model
  -> writes JOB_CREATED + summary_input_segments + INPUT manifest
  -> enqueues advance_job(job_id)

FastAPI
  POST /summary-jobs/upload
  -> writes JOB_CREATED(INGESTING)
  -> streams upload bytes into staging file
  -> writes FILE_STAGED
  -> enqueues ingest_upload(job_id)

FastAPI
  GET /summary-uploads
  -> lists staged uploads that can be reused by new jobs

FastAPI
  POST /summary-jobs/from-upload
  -> writes JOB_CREATED(INGESTING)
  -> references an existing staged upload path
  -> writes FILE_STAGED
  -> enqueues ingest_upload(job_id)

Dramatiq worker
  ingest_upload -> parse staged CSV / Markdown / JSON, write summary_input_segments, write INPUT_READY
  advance_job -> create MAP nodes from input segments
  map_node -> LLM summary per chunk
  reduce_node -> merge summaries by levels
  finalize_job -> final summary

ClickHouse
  source of truth for state, events, artifacts, LLM audit rows
```

## Logging and audit

Logs:

```text
artifacts/summary_backend/logs/summary-backend.log
artifacts/summary_backend/logs/summary-backend.errors.log
```

LLM audit files:

```text
artifacts/summary_backend/audit/{job_id}/{node_id}/{stage}_{call_id}/
  system.txt
  user.txt
  request.json
  response.json
  content.txt
  error.txt
  metadata.json
```

Даже в dry-run режиме пишутся audit files, чтобы проверить структуру трассировки без настоящего LLM.

## Local smoke без очереди и без LLM

Поднять тестовую инфраструктуру:

```bash
docker compose --env-file .env.test-infra.example -f docker-compose.test-infra.yml up -d
```

Запустить синхронный pipeline:

```bash
SUMMARY_BACKEND_DRY_RUN=true python scripts/summary_backend_smoke.py
```

## API + worker

API:

```bash
SUMMARY_BACKEND_DRY_RUN=true python -m summary_backend
```

Worker:

```bash
SUMMARY_BACKEND_DRY_RUN=true dramatiq summary_backend.tasks
```

Frontend:

```bash
SUMMARY_FRONTEND_BACKEND_HTTP_URL=http://localhost:8088 \
SUMMARY_FRONTEND_BACKEND_WS_URL=ws://localhost:8088 \
python -m summary_frontend
```

Whole local stack:

```bash
bash scripts/run_summary_stack.sh
```

Stop local stack:

```bash
bash scripts/stop_summary_stack.sh
```

Создать job:

```bash
curl -sS http://localhost:8088/summary-jobs \
  -H 'Content-Type: application/json' \
  -d '{"input_text":"line 1\nline 2\nline 3","title":"demo"}'
```

Этот endpoint тоже не хранит весь `input_text` как source blob. Он сразу пишет durable rows/chunks в `summary_input_segments`, а `summary_artifacts.INPUT` содержит manifest.

Создать job из файла:

```bash
curl -sS http://localhost:8088/summary-jobs/upload \
  -F 'file=@logs.csv;type=text/csv' \
  -F 'title=logs from dbeaver' \
  -F 'metadata={"source":"manual-upload"}' \
  -F 'source_format=auto' \
  -F 'auto_start=true'
```

Ответ upload endpoint возвращает `status=INGESTING`, `segments_count=0`, `rows_count=0`: файл уже сохранён в staging и поставлен в очередь на ingestion, но строки ещё считает worker. Дальше смотрим:

```text
GET /summary-jobs/{job_id}/events
GET /summary-jobs/{job_id}/input-segments
GET /summary-jobs/{job_id}/snapshot
```

Ожидаемая цепочка событий:

```text
JOB_CREATED INGESTING
FILE_STAGED INGESTING
INPUT_INGEST_STARTED INGESTING
INPUT_INGEST_PROGRESS INGESTING
INPUT_READY INPUT_READY
JOB_RUNNING RUNNING
JOB_DONE DONE
```

Поддерживаемые `source_format`:

| Значение | Что ожидается |
|---|---|
| `auto` | Определение по имени файла и `content-type` |
| `csv` | CSV/TSV с header row; multiline `raw_line` поддерживается через стандартный CSV quoting |
| `markdown` | Markdown pipe table с header и separator row |
| `json` | JSON array или DBeaver-style object, где первое array-значение содержит rows |

Нормализация колонок tolerant к SELECT-ам пользователя:

| Логическое поле | Поддерживаемые имена |
|---|---|
| log text | `raw_line`, `log`, `message`, `line`, `raw`, `text` |
| timestamp | `timestamp`, `time`, `start_time`, `event_time` |
| end time | `end_time`, `finish_time`, `finished_at` |
| namespace | `namespace`, `kubernetes_namespace_name`, `k8s_namespace` |
| container | `container_name`, `kubernetes_container_name`, `container` |
| pod | `pod_name`, `kubernetes_pod_name`, `pod` |

Если log text лежит в нестандартной колонке, передайте form field:

```text
raw_line_column=my_log_column
```

Посмотреть уже загруженные staged files:

```bash
curl -sS 'http://localhost:8088/summary-uploads?limit=100'
```

Создать новую job из уже загруженного файла:

```bash
curl -sS http://localhost:8088/summary-jobs/from-upload \
  -H 'Content-Type: application/json' \
  -d '{
    "upload_id": "job_...",
    "title": "reuse existing upload",
    "metadata": {"source": "manual-reuse"},
    "auto_start": true
  }'
```

`upload_id` сейчас равен `source_job_id` той job, которая впервые staged-файл загрузила. Новая job не копирует файл: она ссылается на тот же staged path и заново пишет свои `summary_input_segments` под новым `job_id`.

Для больших файлов backend не пишет весь upload одним artifact и не парсит его внутри HTTP request. Сначала файл попадает в staging:

```text
artifacts/summary_backend/uploads/{job_id}/{filename}
```

Потом Dramatiq worker парсит staged file, пишет `summary_input_segments`, а `summary_artifacts.INPUT` содержит только manifest: filename, format, row count, segment count. Повторная доставка `ingest_upload` не должна дублировать input: backend сверяется с уже записанными `segment_index`.

Посмотреть input segments:

```text
GET /summary-jobs/{job_id}/input-segments?include_content=false
```

Создать job из ClickHouse query:

```bash
curl -sS http://localhost:8088/summary-jobs/clickhouse-query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "SELECT timestamp, container_name, raw_line FROM logs ORDER BY timestamp LIMIT 1000",
    "title": "logs from clickhouse",
    "metadata": {"source": "manual-query"},
    "auto_start": true
  }'
```

Query source использует отдельные env-переменные. Если они не заданы, берёт основной ClickHouse config:

```text
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_HOST
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_PORT
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_USERNAME
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_PASSWORD
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_DATABASE
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_SECURE
```

Разрешены только `SELECT`/`WITH` read-query. Результат читается блоками через `query_row_block_stream`, нормализуется тем же набором колонок, что и upload.

## Real LLM mode

Dry-run автоматически включён, если нет LLM env. Для реальных вызовов:

```bash
SUMMARY_BACKEND_DRY_RUN=false
SUMMARY_BACKEND_OPENAI_API_BASE=http://llm-gateway.example/v1
SUMMARY_BACKEND_OPENAI_API_KEY=...
SUMMARY_BACKEND_LLM_MODEL='DeepSeek V3.2'
```

Backend использует:

```text
response_format=json_schema
-> message.content
-> json.loads
-> Pydantic validation
```

## Frontend state model

Frontend does not keep pipeline state as source of truth. It stores only the active `job_id` and current form draft in browser `localStorage`. After page refresh it calls:

```text
GET /summary-jobs/{job_id}/snapshot
```

Then it opens:

```text
WS /ws/summary-jobs/{job_id}
```

If WebSocket disconnects, frontend falls back to polling the snapshot endpoint.
