# Summary backend test scenarios

Этот список фиксирует минимальную приёмку backend-а. Автоматические сценарии лежат в `tests/test_summary_backend.py`; integration smoke для реальных ClickHouse/Redis — в `scripts/summary_backend_smoke.py` и `scripts/test_infra_smoke.sh`.

## Автоматические unit-сценарии

| ID | Сценарий | Что проверяет |
|---|---|---|
| SB-U-001 | Full pipeline на fake store/queue/LLM/chunker | `INPUT -> CHUNK -> MAP -> REDUCE levels -> FINAL -> JOB_DONE` |
| SB-U-002 | Несколько REDUCE levels | Динамическое создание уровней reduce до одного результата |
| SB-U-003 | Duplicate node delivery | Повторная доставка node не создаёт второй artifact и не вызывает LLM повторно |
| SB-U-004 | Pause before work | `PAUSE_REQUESTED -> PAUSED`, worker не создаёт nodes |
| SB-U-005 | Resume after pause | `RESUMED` заново ставит `advance_job`, pipeline доходит до `DONE` |
| SB-U-006 | Cancel before work | `CANCEL_REQUESTED -> CANCELLED`, worker не создаёт nodes |
| SB-U-007 | MAP from persisted input segments | Upload/query segments превращаются в CHUNK artifacts без повторного chunking всего input |
| SB-U-008 | CSV parser | DBeaver CSV с multiline `raw_line` нормализуется в log records |
| SB-U-009 | Markdown parser | Markdown pipe table нормализуется в log records |
| SB-U-010 | JSON parser | JSON array и DBeaver object-with-array нормализуются в log records |
| SB-U-011 | Synchronous upload ingestion service | Legacy service создаёт job, manifest artifact, input segments и enqueue |
| SB-U-012 | ClickHouse query source | Query stream rows нормализуются в log records, non-read query rejected |
| SB-U-013 | Staged upload ingestion | Upload staging создаёт `INGESTING`, worker пишет segments, `INPUT_READY`, enqueue advance |
| SB-U-014 | Recovery for ingestion | `recover_jobs()` отправляет `INGESTING` job в `ingest_upload`, остальные runnable jobs в `advance_job` |
| SB-U-015 | Reuse staged upload | Saved upload создаёт новую job без повторной загрузки файла, каталог не показывает reuse-job как новый файл |

Запуск:

```bash
python -m unittest tests.test_summary_backend tests.test_summary_input_files tests.test_summary_query_sources -v
```

## Local integration smoke

| ID | Сценарий | Команда |
|---|---|---|
| SB-I-001 | ClickHouse/Redis healthy, init SQL применился | `bash scripts/test_infra_smoke.sh` |
| SB-I-002 | Dry-run pipeline against ClickHouse without queue | `SUMMARY_BACKEND_DRY_RUN=true python scripts/summary_backend_smoke.py` |
| SB-I-003 | FastAPI creates job with `auto_start=false` | `POST /summary-jobs`, затем `GET /summary-jobs/{job_id}` |
| SB-I-004 | FastAPI + Dramatiq + Redis end-to-end | API создаёт job, worker доводит её до `DONE` |
| SB-I-005 | Staged upload CSV end-to-end | `POST /summary-jobs/upload -> INGESTING`, worker пишет `INPUT_READY`, затем job доходит до `DONE` |
| SB-I-006 | ClickHouse query end-to-end | `POST /summary-jobs/clickhouse-query`, rows пишутся в `input-segments` |
| SB-I-007 | Frontend refresh persistence | `job_id` остаётся в `localStorage`, экран восстанавливается через `/snapshot` |
| SB-I-008 | Frontend WebSocket live updates | Browser получает snapshots через `/ws/summary-jobs/{job_id}` |
| SB-I-009 | Reuse upload end-to-end | `GET /summary-uploads`, затем `POST /summary-jobs/from-upload`, новая job доходит до `DONE` |

## Manual / production-like scenarios

| ID | Сценарий | Ожидаемое поведение |
|---|---|---|
| SB-M-001 | Real LLM structured output | `response_format=json_schema`, затем `json.loads` + Pydantic validation |
| SB-M-002 | Invalid JSON/schema from LLM | Retry, audit files содержат request/response/error |
| SB-M-003 | Rate limit / timeout / 5xx | Retry/backoff, LLM call audit row пишется в ClickHouse |
| SB-M-004 | Context too long | Chunk/reduce group должен быть разделён и переотправлен |
| SB-M-005 | Worker crash after LLM result | Повторная доставка node видит artifact и skip-ает без второго LLM-вызова |
| SB-M-006 | Backend restart | `recover_jobs` находит runnable jobs и продолжает с сохранённых artifacts |
| SB-M-007 | Audit inspection | Для каждого LLM call есть `system.txt`, `user.txt`, `request.json`, `response.json`/`error.txt`, `metadata.json` |

## Что пока не автоматизировано

- реальные ошибки LLM gateway (`429`, `5xx`, timeout);
- split-on-context-overflow;
- запуск настоящего HTTP server + worker как отдельные процессы в CI;
- browser smoke можно запускать локально через Playwright CLI;
- нагрузочный тест на тысячи chunks.

Эти сценарии требуют либо controllable fake LLM HTTP server, либо отдельный integration стенд.
