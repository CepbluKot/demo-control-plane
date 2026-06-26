# Airflow LLM Connectivity Probe

Минимальный DAG для проверки, видит ли Airflow OpenAI-compatible LLM endpoint,
который использует `llm-summary-generator`.

## Файлы

- `dags/summary_llm_connectivity_probe.py` - DAG без расписания, запускается вручную.

## Конфигурация

DAG читает значения из Airflow Variables, а если их нет - из env:

| Airflow Variable / env | Fallback names | Что это |
|---|---|---|
| `SUMMARY_BACKEND_OPENAI_API_BASE` | `OPENAI_API_BASE_DB`, `OPENAI_BASE_URL`, `LLM_API_BASE`, `OPENAI_API_BASE`, `API_BASE` | Base URL LLM gateway. Можно указывать с `/v1` или без него. |
| `SUMMARY_BACKEND_OPENAI_API_KEY` | `OPENAI_API_KEY_DB`, `LLM_API_KEY`, `OPENAI_API_KEY`, `API_KEY` | API key / bearer token. |
| `SUMMARY_BACKEND_LLM_MODEL` | `LLM_MODEL_ID`, `OPENAI_MODEL`, `OPENAI_BIG_MODEL`, `LLM_MODEL`, `MODEL_ID`, `MODEL` | ID модели. |
| `SUMMARY_BACKEND_LLM_TIMEOUT_SECONDS` | `LLM_TIMEOUT_SECONDS` | Timeout HTTP-вызова, по умолчанию `30`. |

Пример установки Variables:

```bash
airflow variables set SUMMARY_BACKEND_OPENAI_API_BASE 'http://llm-gateway.example/v1'
airflow variables set SUMMARY_BACKEND_OPENAI_API_KEY '***'
airflow variables set SUMMARY_BACKEND_LLM_MODEL 'DeepSeek V3.2'
```

## Запуск

Положить содержимое `dags/` в Airflow `dags_folder` или примонтировать эту папку
как DAG source, затем запустить DAG `summary_llm_connectivity_probe` вручную.

Опционально можно переопределить не-secret параметры через trigger config:

```json
{
  "api_base": "http://llm-gateway.example/v1",
  "model": "DeepSeek V3.2",
  "timeout_seconds": 30,
  "max_tokens": 8,
  "prompt": "Reply with OK."
}
```

API key лучше передавать только через Airflow Variable или env: `dag_run.conf`
виден в UI и истории запусков.

Успешный запуск означает, что из Airflow worker есть сетевой доступ до endpoint,
авторизация проходит, и выбранная модель отвечает на `/v1/chat/completions`.
