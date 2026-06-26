# ТЗ на frontend для Summary Jobs

## 1. Цель

Frontend должен быть рабочим интерфейсом для создания, наблюдения и управления задачами суммаризации больших логов.

Backend уже является control plane: он создаёт jobs, принимает входные данные, пишет состояние pipeline в ClickHouse, выполняет работу через Dramatiq/Redis и отдаёт read-model через REST + WebSocket.

Frontend не должен хранить состояние pipeline у себя. Он только создаёт задачи, отображает backend snapshot и отправляет управляющие команды.

## 2. Backend Contract

Базовый URL backend-а задаётся конфигом frontend-а.

Минимальные env frontend-а:

```bash
SUMMARY_FRONTEND_BACKEND_HTTP_URL=http://localhost:8088
SUMMARY_FRONTEND_BACKEND_WS_URL=ws://localhost:8088
```

### Health

```http
GET /health
```

Использовать для индикатора доступности backend-а.

### Создание job из текста

```http
POST /summary-jobs
Content-Type: application/json
```

Request:

```json
{
  "title": "incident-2026-06-23",
  "input_text": "large context",
  "metadata": {
    "source": "frontend"
  },
  "auto_start": true
}
```

Response:

```json
{
  "job_id": "job_...",
  "status": "CREATED",
  "queued": true
}
```

### Создание job из файла

```http
POST /summary-jobs/upload
Content-Type: multipart/form-data
```

Form fields:

| Поле | Тип | Обязательное | Описание |
|---|---|---:|---|
| `file` | file | да | CSV / JSON / Markdown export |
| `title` | string | нет | Название задачи |
| `metadata` | JSON string | нет | Например `{"source":"frontend-upload"}` |
| `auto_start` | boolean | нет | По умолчанию `true` |
| `source_format` | string | нет | `auto`, `csv`, `json`, `markdown` |
| `raw_line_column` | string | нет | Явная колонка с текстом лога, если auto-detect не подходит |

Response:

```json
{
  "job_id": "job_...",
  "status": "INGESTING",
  "queued": true,
  "filename": "logs.csv",
  "source_format": "csv",
  "segments_count": 0,
  "rows_count": 0
}
```

Важно: upload endpoint возвращает `INGESTING`. Это значит, что файл уже сохранён в staging, но строки ещё парсятся worker-ом. Реальные `rows_count` и `segments_count` нужно показывать после события `INPUT_READY` или через `GET /summary-jobs/{job_id}/input-segments`.

### Просмотр уже загруженных файлов

```http
GET /summary-uploads?limit=200
```

Response:

```json
[
  {
    "upload_id": "job_...",
    "source_job_id": "job_...",
    "filename": "logs.csv",
    "source_format": "csv",
    "content_type": "text/csv",
    "raw_line_column": "raw_line",
    "size_bytes": 1024,
    "available": true,
    "job_status": "DONE",
    "staged_at": "..."
  }
]
```

`available=false` значит, что metadata в БД есть, но staged-файл на диске уже недоступен. Такой upload нельзя выбирать для новой job.

### Создание job из уже загруженного файла

```http
POST /summary-jobs/from-upload
Content-Type: application/json
```

Request:

```json
{
  "upload_id": "job_...",
  "title": "reuse logs.csv",
  "metadata": {
    "source": "frontend-reuse"
  },
  "auto_start": true,
  "source_format": null,
  "raw_line_column": null
}
```

Response такой же, как у upload endpoint:

```json
{
  "job_id": "job_...",
  "status": "INGESTING",
  "queued": true,
  "filename": "logs.csv",
  "source_format": "csv",
  "segments_count": 0,
  "rows_count": 0
}
```

Новая job не загружает файл повторно. Backend использует уже существующий staged file path, но input segments пишет заново под новым `job_id`.

### Создание job из ClickHouse query

```http
POST /summary-jobs/clickhouse-query
Content-Type: application/json
```

Request:

```json
{
  "title": "logs query",
  "query": "SELECT timestamp, container_name, raw_line FROM logs ORDER BY timestamp LIMIT 1000",
  "metadata": {
    "source": "frontend-query"
  },
  "auto_start": true,
  "raw_line_column": "raw_line"
}
```

Response:

```json
{
  "job_id": "job_...",
  "status": "CREATED",
  "queued": true,
  "source_format": "clickhouse_query",
  "segments_count": 12,
  "rows_count": 1000
}
```

Backend принимает только read-query: `SELECT` или `WITH`.

### Snapshot

```http
GET /summary-jobs/{job_id}/snapshot
```

Главный endpoint для UI. Его же backend отправляет по WebSocket.

Response shape:

```json
{
  "job": {
    "job_id": "job_...",
    "job_status": "RUNNING",
    "last_event_type": "JOB_RUNNING",
    "updated_at": "...",
    "events_count": 10
  },
  "node_counts": {
    "DONE": 4,
    "RUNNING": 1
  },
  "artifact_counts": {
    "INPUT": 1,
    "MAP_SUMMARY": 4
  },
  "nodes": [],
  "artifacts": [],
  "final_artifact": null,
  "job_events": [],
  "node_events": [],
  "server_time": "..."
}
```

### WebSocket updates

```text
WS /ws/summary-jobs/{job_id}
```

Backend отправляет сообщения:

```json
{
  "type": "snapshot",
  "snapshot": {}
}
```

или:

```json
{
  "type": "error",
  "detail": "job not found: job_..."
}
```

Frontend обязан:

- подключаться к WebSocket после создания или открытия job;
- перерисовывать весь экран из `snapshot`;
- при disconnect/error переходить на polling `GET /summary-jobs/{job_id}/snapshot`;
- после восстановления WebSocket отключать polling;
- не терять job после refresh страницы.

### Управление job

```http
POST /summary-jobs/{job_id}/pause
POST /summary-jobs/{job_id}/resume
POST /summary-jobs/{job_id}/cancel
```

Response:

```json
{
  "job_id": "job_...",
  "status": "PAUSE_REQUESTED"
}
```

После команды frontend не должен сам менять состояние как финальное. Нужно дождаться WebSocket/polling snapshot.

### Дополнительные read endpoints

```http
GET /summary-jobs/{job_id}
GET /summary-jobs/{job_id}/events?limit=500
GET /summary-jobs/{job_id}/node-events?limit=1000
GET /summary-jobs/{job_id}/nodes
GET /summary-jobs/{job_id}/artifacts?include_content=false
GET /summary-jobs/{job_id}/input-segments?include_content=false
```

Использовать для детальных вкладок, модалок и debug views. Основной экран должен жить от `snapshot`.

## 3. Поддерживаемые статусы

Job statuses:

| Статус | UI смысл |
|---|---|
| `CREATED` | Job создана, может быть поставлена в очередь |
| `INGESTING` | Файл сохранён, worker парсит вход и пишет input segments |
| `INPUT_READY` | Входные данные нормализованы, pipeline готов к MAP |
| `RUNNING` | Идёт MAP / REDUCE / FINAL |
| `PAUSE_REQUESTED` | Пользователь запросил pause, backend останавливает дальнейшее продвижение |
| `PAUSED` | Job остановлена |
| `RESUMED` | Resume запрошен, pipeline продолжает работу |
| `CANCEL_REQUESTED` | Пользователь запросил cancel |
| `CANCELLED` | Job отменена |
| `WAITING_RETRY` | Временная ошибка, backend ждёт retry |
| `WAITING_PROVIDER` | LLM/provider недоступен или rate limit |
| `FAILED` | Job завершилась ошибкой |
| `DONE` | Финальный summary готов |

Node statuses:

| Статус | UI смысл |
|---|---|
| `PENDING` | Node создан, но ещё не выполняется |
| `QUEUED` | Node поставлен в очередь |
| `RUNNING` | Выполняется LLM-вызов или обработка |
| `PAUSED` | Остановлен из-за pause |
| `WAITING_RETRY` | Retry после временной ошибки |
| `DONE` | Node завершён |
| `FAILED_RETRYABLE` | Временная ошибка, можно повторять |
| `FAILED_FINAL` | Финальная ошибка node |
| `SKIPPED_ALREADY_DONE` | Повторная доставка actor-а была пропущена как уже выполненная |

## 4. Основные пользовательские сценарии

### 4.1 Создание из текста

Пользователь вводит title, вставляет текст, выбирает `auto_start`, нажимает create.

UI:

- отправляет `POST /summary-jobs`;
- сохраняет `job_id` в localStorage;
- открывает экран job;
- подключает WebSocket;
- показывает pipeline, события и итоговый summary.

### 4.2 Создание из файла

Пользователь выбирает CSV / JSON / Markdown export.

UI должен:

- показывать имя файла и размер до отправки;
- дать выбор `source_format`: auto/csv/json/markdown;
- дать optional поле `raw_line_column`;
- отправлять multipart form;
- после ответа показать `INGESTING`;
- явно отобразить этап ingestion до появления MAP nodes;
- после `INPUT_READY` показать число input segments и rows, если backend вернул это в manifest/events.

Большие файлы:

- не читать файл целиком в JS для preview;
- не пытаться парсить файл в браузере;
- показывать upload progress, если используется `XMLHttpRequest` или streaming upload client;
- после получения `job_id` считать backend source of truth.

### 4.3 Создание из уже загруженного файла

Пользователь открывает режим `Saved Upload`, видит список ранее загруженных staged-файлов и выбирает один из них.

UI должен:

- загрузить список через `GET /summary-uploads?limit=200`;
- показать filename, size, source_format, status исходной job и staged time;
- не разрешать выбрать upload с `available=false`;
- позволить refresh списка без перезагрузки страницы;
- позволить optional override `source_format` и `raw_line_column`;
- отправить `POST /summary-jobs/from-upload`;
- после ответа показать новую job в `INGESTING`;
- подключиться к WebSocket новой job и дальше работать как с обычной upload job.

Важно: saved upload mode не должен повторно отправлять файл с браузера. Frontend передаёт только `upload_id`, а backend переиспользует staged path.

### 4.4 Создание из ClickHouse SQL

Пользователь вводит SQL.

UI должен:

- подсказать, что разрешены только `SELECT`/`WITH`;
- иметь поле `raw_line_column`;
- отправить `POST /summary-jobs/clickhouse-query`;
- показать rows/segments из response;
- открыть job snapshot.

### 4.5 Наблюдение pipeline

Экран job должен показывать:

- общий job status;
- connection state: `live`, `polling`, `offline/error`;
- прогресс nodes: done / total;
- количество artifacts по типам;
- input state: source type, rows, segments;
- список nodes с `node_type`, `level`, `node_index`, `node_status`, `node_id`;
- job events;
- node events;
- final summary.

Для REDUCE уровней показывать level явно: `REDUCE L1`, `REDUCE L2`, ...

### 4.6 Pause / Resume / Cancel

Кнопки:

- `Pause` активна для `INGESTING`, `INPUT_READY`, `RUNNING`, `WAITING_RETRY`, `WAITING_PROVIDER`;
- `Resume` активна для `PAUSED`, `PAUSE_REQUESTED`;
- `Cancel` активна для всех незавершённых статусов.

После клика:

- кнопка временно disabled;
- UI показывает pending action;
- фактический статус берётся только из snapshot.

### 4.7 Refresh страницы

Frontend обязан хранить минимум:

```text
active job_id
draft title
draft input_text
draft query_text
draft raw_line_column values
draft auto_start
последний выбранный input mode
```

После refresh:

- если есть `job_id`, загрузить `GET /summary-jobs/{job_id}/snapshot`;
- восстановить WebSocket;
- если job не найдена, показать ошибку и предложить очистить local state.

## 5. Экраны и компоненты

### 5.1 Shell

- Верхняя панель: название, active job id/title, health/connection badge.
- Основная область: слева создание job, справа наблюдение текущей job.
- Layout должен работать на desktop и tablet; mobile достаточно одноколоночного режима.

### 5.2 Create Job Panel

Обязательные input modes:

- `Text`
- `File Upload`
- `Saved Upload`
- `ClickHouse Query`

Общие поля:

- title;
- metadata JSON editor или key/value минимальный редактор;
- auto_start toggle.

Text mode:

- textarea для input_text;
- счётчик символов.

File mode:

- file picker;
- source_format select;
- raw_line_column input;
- filename/size display;
- upload progress.

Saved Upload mode:

- список из `GET /summary-uploads`;
- refresh button;
- filename, size, source format, original job status, staged time;
- disabled state для `available=false`;
- optional source_format override;
- optional raw_line_column override.

Query mode:

- SQL editor textarea;
- raw_line_column input;
- подсказка про read-only query.

### 5.3 Job Overview

Показывать:

- status pill;
- created/updated/server time;
- progress;
- counts по node statuses;
- counts по artifact types;
- input rows/segments, если доступны.

### 5.4 Pipeline View

Таблица или timeline:

| Поле | Описание |
|---|---|
| node_type | MAP / REDUCE / FINAL |
| level | 0 для MAP, 1+ для REDUCE |
| node_index | индекс node на уровне |
| node_status | текущий статус |
| updated_at | последнее событие |
| node_id | технический id |

Нужны фильтры:

- all;
- running/waiting;
- failed;
- done.

### 5.5 Events View

Показывать job events и node events в одной ленте, сортировать по времени desc.

Для каждой записи:

- scope: job/node;
- event_type;
- status;
- actor;
- message;
- время;
- payload в expandable JSON.

### 5.6 Artifacts View

Показывать artifacts:

- `INPUT`
- `CHUNK`
- `MAP_SUMMARY`
- `REDUCE_SUMMARY`
- `FINAL_SUMMARY`

По умолчанию не грузить `content` для всех artifacts. Для просмотра конкретного artifact использовать отдельную загрузку через `GET /summary-jobs/{job_id}/artifacts?include_content=true` или будущий точечный endpoint, если он будет добавлен.

### 5.7 Input Segments View

Отдельная вкладка для входных сегментов:

- segment_index;
- source_type;
- source_format;
- rows_count;
- chars;
- content_hash;
- metadata.

По умолчанию `include_content=false`. Content открывать по явному действию пользователя.

### 5.8 Final Summary

Если `final_artifact.content` является JSON `SummaryResult`, отрисовать:

- summary;
- key_points;
- warnings;
- source_count.

Если content не JSON, показать plain text.

## 6. Ошибки и надёжность

Frontend должен различать:

- backend unavailable;
- WebSocket disconnected;
- validation error `422`;
- job not found `404`;
- provider waiting/retry статусы в snapshot.

Поведение:

- сетевые ошибки не очищают active job;
- при WebSocket error включается polling;
- при polling error показывается badge, но экран не очищается;
- при `FAILED` показывать последние job/node events и payload ошибки;
- при `WAITING_PROVIDER`/`WAITING_RETRY` показывать, что backend сам продолжит или job можно поставить на pause.

## 7. Нефункциональные требования

- Не хранить большие входные файлы и artifacts в localStorage.
- Не парсить большие файлы в браузере.
- Не делать polling чаще 1-2 секунд.
- Не запрашивать artifact/input segment content массово.
- Все render-операции должны быть идемпотентны: один snapshot полностью заменяет UI state.
- UI должен выдерживать сотни nodes/events без заметных лагов; длинные списки желательно виртуализировать или ограничивать.
- Все длинные id/hash/message должны переноситься и не ломать layout.

## 8. Минимальная приёмка

1. Создание job из текста доходит до `DONE`, final summary отображается.
2. Создание job из CSV upload показывает `INGESTING`, затем pipeline и `DONE`.
3. Создание job из JSON upload работает.
4. Создание job из Markdown upload работает.
5. Saved upload list показывает ранее загруженные файлы из `GET /summary-uploads`.
6. Создание job из saved upload не загружает файл повторно и доходит до `DONE`.
7. Reuse-job не появляется в saved upload list как новый файл.
8. Создание job из ClickHouse query показывает rows/segments и запускает pipeline.
9. WebSocket обновляет экран без ручного refresh.
10. При отключении WebSocket frontend переходит на polling.
11. Refresh страницы восстанавливает active job.
12. Pause останавливает продвижение pipeline, Resume продолжает.
13. Cancel переводит job в terminal state.
14. При `FAILED` видны последние события и payload ошибки.
15. Большой upload не блокирует UI; после ответа backend-а job остаётся отслеживаемой.

## 9. Что сейчас уже есть в reference frontend

В `summary_frontend` уже есть минимальный static frontend:

- создание job из текста;
- создание job через file upload;
- создание job из сохранённого upload;
- создание job из ClickHouse query;
- localStorage для active job, выбранного input mode и draft-полей;
- загрузка saved upload list через `GET /summary-uploads`;
- WebSocket `/ws/summary-jobs/{job_id}`;
- fallback polling;
- отображение nodes, artifacts, events и final summary;
- pause/resume/cancel.

Не хватает для полного ТЗ:

- input segments view;
- expandable event payload;
- artifact content viewer;
- upload progress;
- более детального отображения ingestion стадии.
