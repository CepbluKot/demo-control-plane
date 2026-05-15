"""Тригерит log_summarizer DAG через Airflow REST API.

Настрой параметры в секции CONFIG и запусти:
    python airflow/trigger_dag.py
"""
import json
import sys
import urllib.request
import urllib.error
from base64 import b64encode


# ── CONFIG ────────────────────────────────────────────────────────────────────

AIRFLOW_URL      = "http://localhost:8080"
AIRFLOW_USERNAME = "airflow"
AIRFLOW_PASSWORD = "airflow"

INCIDENT_CONTEXT = "Airflow: массовые ошибки Pod creation failed (Forbidden) в kubernetes_executor."
PERIOD_START     = "2026-03-18T01:00:00"   # UTC
PERIOD_END       = "2026-03-18T04:00:00"   # UTC

OUTPUT_PATH          = "/data/report.md"
CONTEXT_TOKENS       = 150000
MAP_CONCURRENCY      = 5
BATCH_SIZE           = 1000
MAX_EVENTS_PER_MERGE = 30
TOOL_CALLING         = False

# ── SQL ───────────────────────────────────────────────────────────────────────

LOGS_SQL = """
SELECT
    start_time                        AS timestamp,
    end_time,
    ''                                AS namespace,
    container_name,
    pod_name,
    ''                                AS image_tag,
    concat(
        '[', toString(start_time),
        ' → ', toString(end_time), ']',
        ' ×', toString(cnt),
        '  ', pod_name, '/', container_name,
        '  ', log_text
    )                                 AS raw_line
FROM (
    SELECT
        min(timestamp)  AS start_time,
        max(timestamp)  AS end_time,
        min(message)    AS log_text,
        count()         AS cnt,
        any(container)  AS container_name,
        any(pod)        AS pod_name
    FROM (
        SELECT *,
            sum(is_new_group) OVER (
                PARTITION BY container, pod
                ORDER BY timestamp ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS group_id
        FROM (
            SELECT *,
                if(
                    right(message, 10) != ifNull(
                        lagInFrame(right(message, 10)) OVER (
                            PARTITION BY container, pod
                            ORDER BY timestamp ASC
                        ), ''
                    ),
                    1, 0
                ) AS is_new_group
            FROM shards.`logs_k8s_k-ndp-v01-dadm-streaming`
            WHERE timestamp >  parseDateTime64BestEffort('{last_ts}')
              AND timestamp <= parseDateTime64BestEffort('{period_end}')
              AND cluster = 'ndp-v01'
              AND multiSearchAny(lower(message), [
                    'fatal', 'critical', 'error', 'exception',
                    'failed', 'failure', 'crash', 'timeout',
                    'out of memory', 'oom', 'killed', 'traceback'
                ])
            ORDER BY timestamp ASC
            LIMIT {raw_limit}
        )
    )
    GROUP BY group_id, container, pod
)
ORDER BY start_time ASC
LIMIT {limit}
"""

METRICS_SQL = None   # строка с SQL или None если метрики не нужны

# ── RUN ───────────────────────────────────────────────────────────────────────

def main() -> None:
    url = f"{AIRFLOW_URL.rstrip('/')}/api/v1/dags/log_summarizer/dagRuns"

    payload = {
        "conf": {
            "incident_context":     INCIDENT_CONTEXT,
            "start":                PERIOD_START,
            "end":                  PERIOD_END,
            "logs_sql":             LOGS_SQL.strip(),
            "metrics_sql":          METRICS_SQL,
            "output_path":          OUTPUT_PATH,
            "context_tokens":       CONTEXT_TOKENS,
            "map_concurrency":      MAP_CONCURRENCY,
            "batch_size":           BATCH_SIZE,
            "max_events_per_merge": MAX_EVENTS_PER_MERGE,
            "tool_calling":         TOOL_CALLING,
        }
    }

    token = b64encode(f"{AIRFLOW_USERNAME}:{AIRFLOW_PASSWORD}".encode()).decode()
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
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)

    print(f"dag_run_id:   {result['dag_run_id']}")
    print(f"state:        {result['state']}")
    print(f"logical_date: {result.get('logical_date', '—')}")


if __name__ == "__main__":
    main()
