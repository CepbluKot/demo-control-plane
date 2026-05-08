"""
Airflow DAG: Log Summarizer (incident analysis)

Запускает LLM-пайплайн анализа инцидентов в Docker-контейнере.
Читает логи из ClickHouse, применяет MAP-REDUCE через LLM,
записывает Markdown-отчёт в примонтированный volume /data.

Airflow Variables (задать в Admin → Variables):
  LLM_API_BASE  — например http://llm-server:8000
  LLM_API_KEY   — API ключ
  LLM_MODEL     — название модели
  CH_HOST       — хост ClickHouse
  CH_PORT       — HTTP-порт ClickHouse (обычно 8123)
  CH_USER       — пользователь ClickHouse
  CH_PASSWORD   — пароль ClickHouse
  CH_DATABASE   — база данных ClickHouse

Params (задаются при ручном триггере или в dag_run.conf):
  incident_context     — описание инцидента в свободной форме
  start                — начало инцидента ISO8601 (напр. 2024-01-15T14:00:00)
  end                  — конец инцидента ISO8601
  logs_sql             — SQL-шаблон или путь к .sql-файлу в /data
                         Плейсхолдеры: {last_ts}, {period_end}, {limit}, {raw_limit}
  metrics_sql          — SQL-шаблон для метрик (опционально)
  output_path          — путь для отчёта внутри контейнера (/data/...)
  context_tokens       — размер контекста модели в токенах (default 150000)
  map_concurrency      — параллельность MAP (default 5)
  batch_size           — строк ClickHouse на страницу (default 1000)
  max_events_per_merge — макс. событий при слиянии REDUCE (default 30)
  max_reduce_rounds    — макс. раундов REDUCE (default 15)
  tool_calling         — включить TOOLS mode в instructor (default false)
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

DATA_DIR = "/opt/airflow/data"  # host-путь, монтируется в /data внутри контейнера
RUNS_DIR = "/opt/airflow/runs"  # host-путь для артефактов runs/

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="log_summarizer",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["log-summarizer", "incidents", "llm"],
    params={
        "incident_context": Param(
            "",
            type="string",
            description="Описание инцидента (напр. 'payments OOM at 14:30 UTC')",
        ),
        "start": Param(
            "",
            type="string",
            description="Начало инцидента ISO8601, напр. 2024-01-15T14:00:00",
        ),
        "end": Param(
            "",
            type="string",
            description="Конец инцидента ISO8601, напр. 2024-01-15T15:00:00",
        ),
        "logs_sql": Param(
            "/data/logs.sql",
            type="string",
            description=(
                "SQL-шаблон или путь к .sql-файлу в /data. "
                "Плейсхолдеры: {last_ts}, {period_end}, {limit}, {raw_limit}. "
                "Колонки: timestamp, end_time, raw_line."
            ),
        ),
        "metrics_sql": Param(
            "",
            type="string",
            description="SQL-шаблон для метрик (опционально). Путь к .sql или inline SQL.",
        ),
        "output_path": Param(
            "/data/report.md",
            type="string",
            description="Путь для записи Markdown-отчёта внутри контейнера",
        ),
        "context_tokens": Param(
            150000,
            type="integer",
            minimum=4000,
            description="Размер контекстного окна модели в токенах",
        ),
        "map_concurrency": Param(5, type="integer", minimum=1, maximum=20),
        "batch_size": Param(
            1000,
            type="integer",
            minimum=100,
            description="Строк ClickHouse на страницу",
        ),
        "max_events_per_merge": Param(30, type="integer", minimum=5),
        "max_reduce_rounds": Param(15, type="integer", minimum=1),
        "tool_calling": Param(
            False,
            type="boolean",
            description="TOOLS mode в instructor (отключить для vLLM без tool calling)",
        ),
    },
) as dag:

    run_log_summarizer = DockerOperator(
        task_id="run_log_summarizer",
        image="log-summarizer:latest",
        command=(
            "--context         '{{ params.incident_context }}'"
            " --start          {{ params.start }}"
            " --end            {{ params.end }}"
            " --output         {{ params.output_path }}"
            " --context-tokens {{ params.context_tokens }}"
            " --map-concurrency {{ params.map_concurrency }}"
            " --batch-size     {{ params.batch_size }}"
            " --max-events-per-merge {{ params.max_events_per_merge }}"
            " --max-reduce-rounds {{ params.max_reduce_rounds }}"
            " {% if params.tool_calling %}--tool-calling{% endif %}"
        ),
        environment={
            "LLM_API_BASE": "{{ var.value.LLM_API_BASE }}",
            "LLM_API_KEY":  "{{ var.value.LLM_API_KEY }}",
            "LLM_MODEL":    "{{ var.value.LLM_MODEL }}",
            "CH_HOST":      "{{ var.value.CH_HOST }}",
            "CH_PORT":      "{{ var.value.CH_PORT }}",
            "CH_USER":      "{{ var.value.CH_USER }}",
            "CH_PASSWORD":  "{{ var.value.CH_PASSWORD }}",
            "CH_DATABASE":  "{{ var.value.CH_DATABASE }}",
            # SQL передаём через env — избегаем проблем с экранированием спецсимволов в CLI
            "LOGS_SQL":    "{{ params.logs_sql }}",
            "METRICS_SQL": "{{ params.metrics_sql }}",
        },
        mounts=[
            Mount(source=DATA_DIR, target="/data",     type="bind"),
            Mount(source=RUNS_DIR, target="/app/runs", type="bind"),
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove="success",
        mount_tmp_dir=False,
    )
