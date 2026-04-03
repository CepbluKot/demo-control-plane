# Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Настройка `.env`

```bash
cp .env.example .env
```

## Описание ключевых параметров `.env`

| Параметр | Что делает |
|---|---|
| `OPENAI_API_BASE_DB` | URL LLM API. |
| `OPENAI_API_KEY_DB` | API-ключ для LLM. |
| `LLM_MODEL_ID` | ID модели для LLM-вызовов. |
| `CONTROL_PLANE_LOGS_CLICKHOUSE_HOST` / `PORT` / `USERNAME` / `PASSWORD` | Подключение к ClickHouse с логами. |
| `CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY` | SQL-шаблон для логов (`{period_start}`, `{period_end}`, `{limit}`, `{offset}`, `{last_ts}`). |
| `CONTROL_PLANE_LOGS_PAGE_LIMIT` | Размер страницы чтения из БД. |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_DB_BATCH_SIZE` | Размер DB-батча на странице `Logs Summarizer`. |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE` | Максимум строк на один MAP-вызов LLM. |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_MIN_LLM_BATCH_SIZE` | Минимальный размер LLM-батча при авто-уменьшении. |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_AUTO_SHRINK_ON_400` | Авто-уменьшение батча при overflow/400. |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_TIMEOUT` | Базовый timeout LLM-вызовов (сек). |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_RETRIES` | Количество retry (`-1` = бесконечно). |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_FINAL_STAGE_MAX_RETRIES` | Retry для финального этапа отчёта. |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_FINAL_STAGE_LLM_TIMEOUT` | Timeout финального этапа отчёта (сек). |
| `CONTROL_PLANE_UI_LOGS_SUMMARY_FINAL_STAGE_CONTEXT_MAX_CHARS` | Лимит контекста для LLM на финальном этапе (сжатие только для prompt, в файлы сохраняется полный отчёт). |
| `CONTROL_PLANE_LLM_USE_INSTRUCTOR` | Включает structured-вызовы через Instructor/Pydantic. |
| `CONTROL_PLANE_LLM_SUPPORTS_TOOL_CALLING` | `false`, если ваш gateway/model не поддерживает tool-calling. |
| `CONTROL_PLANE_LLM_MAP_PROMPT_TEMPLATE` / `REDUCE` / `FREEFORM` / `UI_FINAL_REPORT` | Кастомные шаблоны промптов. |

Полный список параметров и примеры значений смотрите в `.env.example`.
