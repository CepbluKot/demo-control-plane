"""
Airflow DAG: Log Summarizer (incident analysis)

Запускает LLM-пайплайн анализа инцидентов как Pod в Kubernetes.
Читает логи из ClickHouse, применяет MAP-REDUCE через LLM,
записывает Markdown-отчёт в примонтированный PVC /data.

Airflow Variables (Admin → Variables):
  LLM_API_BASE  — например http://llm-server:8000
  LLM_API_KEY   — API ключ
  LLM_MODEL     — название модели
  CLICKHOUSE_MONLOG_HOST     — хост ClickHouse
  CLICKHOUSE_MONLOG_PORT     — HTTP-порт ClickHouse (обычно 8123)
  CLICKHOUSE_MONLOG_USER     — пользователь ClickHouse
  CLICKHOUSE_MONLOG_PASSWORD — пароль ClickHouse
  CH_DATABASE                — база данных ClickHouse
  LOGS_SQL      — SQL-шаблон для логов (многострочный — редактировать здесь)
  METRICS_SQL   — SQL-шаблон для метрик (опционально)
  LOG_SUMMARIZER_IMAGE     — образ (default: registry.your-company.com/log-summarizer:latest)
  LOG_SUMMARIZER_NAMESPACE — namespace для подов (default: airflow)
  LOG_SUMMARIZER_DATA_PVC  — PVC с данными (default: log-summarizer-data)
  LOG_SUMMARIZER_RUNS_PVC  — PVC для артефактов (default: log-summarizer-runs)

Params (задаются при ручном триггере):
  incident_context     — описание инцидента в свободной форме
  start                — начало инцидента ISO8601 (напр. 2024-01-15T14:00:00)
  end                  — конец инцидента ISO8601
  logs_sql             — SQL-шаблон для логов
  metrics_sql          — SQL-шаблон для метрик (опционально)
  output_path          — путь для отчёта внутри пода (/data/...)
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
from airflow.models import Param, Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# ── CONFIG ────────────────────────────────────────────────────────────────────

IMAGE     = Variable.get("LOG_SUMMARIZER_IMAGE",     default_var="registry.your-company.com/log-summarizer:latest")
NAMESPACE = Variable.get("LOG_SUMMARIZER_NAMESPACE", default_var="k-ndp-v01-ndp-monitor-clickhouse-full-benchmark")
DATA_PVC  = Variable.get("LOG_SUMMARIZER_DATA_PVC",  default_var="log-summarizer-data")
RUNS_PVC  = Variable.get("LOG_SUMMARIZER_RUNS_PVC",  default_var="log-summarizer-runs")

# ── DAG ───────────────────────────────────────────────────────────────────────

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
    tags=["log-summarizer", "incidents", "llm", "k8s"],
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
            "",
            type="string",
            description="SQL-шаблон для логов. Плейсхолдеры: {last_ts}, {period_end}, {limit}, {raw_limit}.",
        ),
        "metrics_sql": Param(
            None,
            type=["null", "string"],
            description="SQL-шаблон для метрик (опционально).",
        ),
        "output_path": Param(
            "/data/report.md",
            type="string",
            description="Путь для записи Markdown-отчёта внутри пода",
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

    run_log_summarizer = KubernetesPodOperator(
        task_id="run_log_summarizer",
        name="log-summarizer",
        namespace=NAMESPACE,
        image=IMAGE,
        image_pull_policy="Always",

        arguments=[
            "--context",              "{{ params.incident_context }}",
            "--start",                "{{ params.start }}",
            "--end",                  "{{ params.end }}",
            "--output",               "{{ params.output_path }}",
            "--context-tokens",       "{{ params.context_tokens | string }}",
            "--map-concurrency",      "{{ params.map_concurrency | string }}",
            "--batch-size",           "{{ params.batch_size | string }}",
            "--max-events-per-merge", "{{ params.max_events_per_merge | string }}",
            "--max-reduce-rounds",    "{{ params.max_reduce_rounds | string }}",
        ],

        env_vars={
            "LLM_API_BASE":    "{{ var.value.LLM_API_BASE }}",
            "LLM_API_KEY":     "{{ var.value.LLM_API_KEY }}",
            "LLM_MODEL":       "{{ var.value.LLM_MODEL }}",
            "CH_HOST":         "{{ var.value.CLICKHOUSE_MONLOG_HOST }}",
            "CH_PORT":         "{{ var.value.CLICKHOUSE_MONLOG_PORT }}",
            "CH_USER":         "{{ var.value.CLICKHOUSE_MONLOG_USER }}",
            "CH_PASSWORD":     "{{ var.value.CLICKHOUSE_MONLOG_PASSWORD }}",
            "CH_DATABASE":     "{{ var.value.CH_DATABASE }}",
            "LLM_TOOL_CALLING": "{{ 'true' if params.tool_calling else 'false' }}",
            "LOGS_SQL":        "{{ params.logs_sql }}",
            "METRICS_SQL":     "{{ params.metrics_sql or '' }}",
        },

        security_context=k8s.V1PodSecurityContext(
            run_as_non_root=True,
            run_as_user=1000,
        ),
        container_security_context=k8s.V1SecurityContext(
            read_only_root_filesystem=True,
            run_as_non_root=True,
            run_as_user=1000,
        ),

        volumes=[
            k8s.V1Volume(
                name="data",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=DATA_PVC),
            ),
            k8s.V1Volume(
                name="runs",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=RUNS_PVC),
            ),
            k8s.V1Volume(
                name="tmp",
                empty_dir=k8s.V1EmptyDirVolumeSource(),
            ),
        ],

        volume_mounts=[
            k8s.V1VolumeMount(name="data", mount_path="/data"),
            k8s.V1VolumeMount(name="runs", mount_path="/app/runs"),
            k8s.V1VolumeMount(name="tmp",  mount_path="/tmp"),
        ],

        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "512Mi"},
            limits={"cpu": "2",     "memory": "2Gi"},
        ),

        get_logs=True,
        is_delete_operator_pod=True,
        in_cluster=True,
    )
