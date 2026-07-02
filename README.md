# LLM Summary Generator

Backend + frontend для устойчивого map/reduce summarization pipeline по большим логам.

## Документы

- [Архитектура summary job pipeline](docs/summary-job-architecture.md)
- [Summary backend](docs/summary-backend.md)
- [ТЗ на frontend](docs/summary-frontend-tz.md)
- [Summary backend test scenarios](docs/summary-backend-test-scenarios.md)
- [Тестовая инфраструктура](docs/test-infra.md)

## Локальный запуск

```bash
python3 -m venv .venv-summary-backend
source .venv-summary-backend/bin/activate
pip install -r requirements.txt
```

Поднять Redis + ClickHouse:

```bash
docker compose --env-file .env.test-infra.example -f docker-compose.test-infra.yml up -d
```

Запустить backend, frontend и worker:

```bash
bash scripts/run_summary_stack.sh
```

Frontend: `http://localhost:8090`

Backend: `http://localhost:8088`

## Monitoring Profiles + Dify Scheduler

`summary_backend` теперь умеет работать как control-plane для периодических Dify workflow:

- хранит `monitoring profiles` в ClickHouse;
- хранит per-profile `schedule` с `cron`, `timezone` и `max_active_runs`;
- создает `monitoring runs`;
- вызывает Dify как executor в `response_mode=blocking`;
- может запускать due profiles через ручной tick или фоновый scheduler-loop.

Основные ручки:

- `POST /monitoring-profiles`
- `GET /monitoring-profiles`
- `GET /monitoring-profiles/{profile_id}`
- `POST /monitoring-profiles/{profile_id}/archive`
- `POST /monitoring-profiles/{profile_id}/run`
- `GET /monitoring-runs`
- `GET /monitoring-runs/{run_id}`
- `POST /monitoring-scheduler/tick`

Новые env:

- `SUMMARY_BACKEND_DIFY_API_BASE`
- `SUMMARY_BACKEND_DIFY_TIMEOUT_SECONDS`
- `SUMMARY_BACKEND_DIFY_SECRETS_PATH`
- `SUMMARY_BACKEND_MONITORING_SCHEDULER_ENABLED`
- `SUMMARY_BACKEND_MONITORING_SCHEDULER_POLL_INTERVAL_SECONDS`

Dify API keys резолвятся либо из env, либо из `secrets.local.json`. Для известных workflow используются те же ключи, что и во frontend:

- `DIFY_API_KEY_PA_INLINE_POINTS_PREDICTION_PIPELINE`
- `DIFY_API_KEY_PA_ANOMALY_DETECTION_PIPELINE`
- `DIFY_API_KEY_PA_GENERAL_SUMMARY_GENERATOR`
- `DIFY_API_KEY_PA_CONTROL_PLANE_MAIN_ORCHESTRATOR`
- `DIFY_API_KEY_PA_LOCAL_ORCHESTRATOR`

Остановка:

```bash
bash scripts/stop_summary_stack.sh
```

## Конфигурация

```bash
cp .env.example .env
```

`docker-compose.local.yml` читает базовые значения из `.env.example` и поверх них,
если файл существует, подхватывает локальный `.env`. Туда удобно класть multi-LLM
профили и секреты.

Ключевые группы настроек:

| Параметры | Что делают |
|---|---|
| `SUMMARY_BACKEND_CLICKHOUSE_*` | ClickHouse-хранилище jobs, events, nodes, artifacts и input segments. |
| `SUMMARY_BACKEND_SOURCE_CLICKHOUSE_*` | Источник логов для пользовательских SQL-запросов. Если не заданы, используются `SUMMARY_BACKEND_CLICKHOUSE_*`. |
| `SUMMARY_BACKEND_BROKER_URL` | Redis broker для Dramatiq. |
| `SUMMARY_BACKEND_OPENAI_API_BASE` / `SUMMARY_BACKEND_OPENAI_API_KEY` / `SUMMARY_BACKEND_LLM_MODEL` / `SUMMARY_BACKEND_LLM_MODELS` | Legacy single-LLM конфиг для одного gateway. |
| `SUMMARY_BACKEND_LLM_PROFILES` / `SUMMARY_BACKEND_LLM_PROFILE_DEFAULT` / `SUMMARY_BACKEND_LLM_PROFILE__<ID>__*` | Multi-LLM конфиг: несколько профилей с собственными `api_base`, `api_key` и наборами моделей. |
| `SUMMARY_BACKEND_DRY_RUN` | Тестовый режим без реальных LLM-вызовов. |
| `SUMMARY_BACKEND_UPLOAD_STAGING_DIR` | Staging-директория для файлов, загруженных через `/summary-jobs/upload`. |
| `SUMMARY_FRONTEND_*` | Порт frontend-а и адрес backend-а. |

Если env для LLM не заданы, backend может использовать локальный gitignored fallback-конфиг
`summary_backend/llm_gateway_defaults.json`, который синхронизирован с `../llm_probe.py`.

## Проверки

```bash
SUMMARY_BACKEND_LOG_DIR=/tmp/demo-control-plane-test-logs \
SUMMARY_BACKEND_AUDIT_DIR=/tmp/demo-control-plane-test-audit \
SUMMARY_BACKEND_UPLOAD_STAGING_DIR=/tmp/demo-control-plane-test-uploads \
python -m unittest tests.test_monitoring_control_plane -v
python -m unittest tests.test_summary_backend tests.test_summary_input_files tests.test_summary_query_sources tests.test_summary_frontend -v
python -m compileall -q summary_backend summary_frontend tests scripts
bash scripts/test_infra_smoke.sh
SUMMARY_BACKEND_DRY_RUN=true python scripts/summary_backend_smoke.py --repeat 5
```
