"""Dify Code Node: Trigger log_summarizer Airflow DAG

Запускает log_summarizer DAG через Airflow REST API.

Inputs:
  airflow_url      (str) — базовый URL Airflow, например "http://airflow:8080"
  airflow_username (str) — логин Airflow (Basic Auth)
  airflow_password (str) — пароль Airflow (Basic Auth)
  incident_context (str) — описание инцидента для LLM
  period_start     (str) — начало периода, ISO8601 UTC ("2026-03-18T01:00:00")
  period_end       (str) — конец периода, ISO8601 UTC ("2026-03-18T04:00:00")
  logs_sql         (str) — SQL-запрос для выборки логов (с плейсхолдерами {last_ts}, {period_end}, {limit}, {raw_limit})
  metrics_sql      (str) — SQL для метрик (пустая строка = не использовать)
  output_path      (str) — путь для сохранения отчёта внутри Pod (default "/data/report.md")
  context_tokens   (str) — токенный контекст LLM (default "150000")
  map_concurrency  (str) — параллельных MAP-воркеров (default "5")
  batch_size       (str) — строк логов на один батч (default "1000")
  max_events_per_merge (str) — событий на одну редукцию (default "30")

Outputs:
  dag_run_id   (str) — ID запущенного dag run
  state        (str) — начальное состояние ("queued")
  logical_date (str) — логическая дата запуска
  error        (str) — текст ошибки (пустая строка если успех)
"""

import json
import urllib.request
import urllib.error
from base64 import b64encode


def main(
    airflow_url: str,
    airflow_username: str,
    airflow_password: str,
    incident_context: str,
    period_start: str,
    period_end: str,
    logs_sql: str,
    metrics_sql: str = "",
    output_path: str = "/data/report.md",
    context_tokens: str = "150000",
    map_concurrency: str = "5",
    batch_size: str = "1000",
    max_events_per_merge: str = "30",
) -> dict:
    url = f"{airflow_url.rstrip('/')}/api/v1/dags/log_summarizer/dagRuns"

    payload = {
        "conf": {
            "incident_context":     incident_context,
            "start":                period_start,
            "end":                  period_end,
            "logs_sql":             logs_sql.strip(),
            "metrics_sql":          metrics_sql.strip() or None,
            "output_path":          output_path,
            "context_tokens":       int(context_tokens),
            "map_concurrency":      int(map_concurrency),
            "batch_size":           int(batch_size),
            "max_events_per_merge": int(max_events_per_merge),
        }
    }

    token = b64encode(f"{airflow_username}:{airflow_password}".encode()).decode()
    body  = json.dumps(payload).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Basic {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        return {
            "dag_run_id":   result.get("dag_run_id", ""),
            "state":        result.get("state", ""),
            "logical_date": result.get("logical_date", ""),
            "error":        "",
        }
    except urllib.error.HTTPError as e:
        return {
            "dag_run_id":   "",
            "state":        "error",
            "logical_date": "",
            "error":        f"HTTP {e.code}: {e.read().decode()}",
        }
    except Exception as e:
        return {
            "dag_run_id":   "",
            "state":        "error",
            "logical_date": "",
            "error":        str(e),
        }
