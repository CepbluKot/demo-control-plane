# Тестовая инфраструктура

Локальная инфраструктура нужна, чтобы проверять будущий summary job pipeline без Airflow и без внешних managed-сервисов.

## Состав

```text
ClickHouse 24.8
Redis 7.2
```

ClickHouse хранит append-only events/artifacts/LLM audit. Redis используется как broker для Dramatiq workers.

## Запуск

```bash
docker compose --env-file .env.test-infra.example -f docker-compose.test-infra.yml up -d
```

Проверка:

```bash
bash scripts/test_infra_smoke.sh
```

Остановка:

```bash
docker compose -f docker-compose.test-infra.yml down
```

Полная очистка данных:

```bash
docker compose -f docker-compose.test-infra.yml down -v
```

## Порты

| Сервис | URL |
|---|---|
| ClickHouse HTTP | `http://localhost:8123` |
| ClickHouse native | `localhost:9000` |
| Redis | `redis://localhost:6379/0` |

Upload staging по умолчанию:

```text
artifacts/summary_backend/uploads
```

Переопределяется через:

```text
SUMMARY_BACKEND_UPLOAD_STAGING_DIR
```

Для query ingestion source ClickHouse по умолчанию совпадает с backend ClickHouse. В проде источник логов можно развести отдельными env:

```text
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_HOST
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_PORT
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_USERNAME
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_PASSWORD
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_DATABASE
SUMMARY_BACKEND_SOURCE_CLICKHOUSE_SECURE
```

## ClickHouse schema

Init SQL находится в `infra/clickhouse/init/001_summary_pipeline.sql`.

Создаются таблицы:

| Таблица | Назначение |
|---|---|
| `summary_test.summary_job_events` | События job |
| `summary_test.summary_node_events` | События MAP/REDUCE/FINAL node |
| `summary_test.summary_artifacts` | Input, chunk, MAP, REDUCE, FINAL artifacts |
| `summary_test.summary_input_segments` | Нормализованные сегменты входных файлов/query |
| `summary_test.summary_llm_calls` | Audit LLM-вызовов |

И views:

| View | Назначение |
|---|---|
| `summary_test.summary_job_current_v` | Последний статус job |
| `summary_test.summary_node_current_v` | Последний статус node |

Для ClickHouse 24.8 JSON payload хранится как `String`, а важные поля вынесены в typed columns.
