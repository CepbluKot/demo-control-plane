"""Dify Code Node: Load & Chunk

Copy-paste в Dify Code Node (Python).
Inputs:  period_start (str), period_end (str), incident_info (str), alerts (str)
Outputs: batches (Array[String]), batch_count (Number)

Env vars: CH_HOST, CH_PORT, CH_USER, CH_PASSWORD, CH_DATABASE,
          BATCH_SIZE (default 200), CHUNK_TOKEN_BUDGET (default 6000)
"""
import json
import os
import urllib.parse
import urllib.request


# ── ClickHouse ────────────────────────────────────────────────────────

def ch_query(host, port, user, password, sql, timeout=120):
    encoded = urllib.parse.quote(sql + " FORMAT JSONEachRow")
    url = f"http://{host}:{port}/?query={encoded}"
    req = urllib.request.Request(url)
    req.add_header("X-ClickHouse-User", user)
    req.add_header("X-ClickHouse-Key", password)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        lines = resp.read().decode("utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


# ── Chunker ───────────────────────────────────────────────────────────

def estimate_tokens(text):
    return max(1, len(text) // 4)


def chunk_rows(rows, token_budget=6000):
    if not rows:
        return []
    chunks, current, current_tokens = [], [], 0
    for row in rows:
        t = estimate_tokens(row)
        if current and current_tokens + t > token_budget:
            chunks.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += t
    if current:
        chunks.append("\n".join(current))
    return chunks


# ── SQL ───────────────────────────────────────────────────────────────

CONTAINERS_SQL = """
SELECT
    start_time AS timestamp,
    end_time,
    concat(
        '[', toString(start_time), ' \u2192 ', toString(end_time), ']',
        ' \xd7', toString(cnt),
        '  ', namespace, '/', pod_name,
        '  ', log_text
    ) AS raw_line
FROM (
    SELECT
        min(timestamp) AS start_time, max(timestamp) AS end_time,
        min(log) AS log_text, count() AS cnt,
        any(kubernetes_namespace_name) AS namespace,
        any(kubernetes_pod_name) AS pod_name
    FROM (
        SELECT *,
            sum(is_new_group) OVER (
                PARTITION BY kubernetes_namespace_name, kubernetes_container_name
                ORDER BY timestamp ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS group_id
        FROM (
            SELECT *,
                if(
                    right(log, 10) != ifNull(lagInFrame(right(log, 10)) OVER (
                        PARTITION BY kubernetes_namespace_name, kubernetes_container_name
                        ORDER BY timestamp ASC
                    ), ''), 1, 0
                ) AS is_new_group
            FROM {database}.log_k8s_containers_MT
            WHERE timestamp > parseDateTime64BestEffort('{last_ts}')
              AND timestamp <= parseDateTime64BestEffort('{period_end}')
              AND ext_ClusterName = 'ndp-p01'
              AND (
                    kubernetes_container_name LIKE '%airflow%'
                    OR kubernetes_container_name LIKE '%spark%'
                    OR kubernetes_container_name LIKE '%flex%'
                    OR (kubernetes_namespace_name LIKE '%kube-system%'
                        AND kubernetes_container_name NOT LIKE '%kube-apiserver%')
              )
              AND multiSearchAny(lower(log), [
                    'fatal','critical','error','exception','alert','panic',
                    'failed','failure','crash','abort','timeout','timed out',
                    'deadlock','out of memory','oom','disk full','no space left',
                    'permission denied','access denied','unauthorized','forbidden',
                    'connection refused','connection reset','ssl error','segfault',
                    'killed','rollback','traceback','stack trace'
              ])
            ORDER BY timestamp ASC
        )
    )
    GROUP BY group_id, kubernetes_namespace_name, kubernetes_container_name
)
ORDER BY start_time ASC
LIMIT {limit}
"""

EVENTS_SQL = """
SELECT
    timestamp,
    end_time,
    concat(
        '[EVT:', reason, ']  ',
        '[', toString(timestamp), ']',
        '  ', namespace, '  ', object_name,
        '  ', message
    ) AS raw_line
FROM (
    SELECT
        min(timestamp) AS timestamp, max(timestamp) AS end_time,
        any(reason) AS reason,
        any(kubernetes_namespace_name) AS namespace,
        any(object_name) AS object_name,
        any(message) AS message
    FROM {database}.log_k8s_events
    WHERE timestamp > parseDateTime64BestEffort('{last_ts}')
      AND timestamp <= parseDateTime64BestEffort('{period_end}')
      AND ext_ClusterName = 'ndp-p01'
      AND reason IN (
            'BackOff','ImagePullBackOff','OOMKilling','Evicted','Failed',
            'FailedCreate','FailedScheduling','FailedMount','Killing',
            'NodeNotReady','Unhealthy','CrashLoopBackOff'
      )
    GROUP BY timestamp, kubernetes_namespace_name, object_name
)
ORDER BY timestamp ASC
LIMIT {limit}
"""


# ── Main ──────────────────────────────────────────────────────────────

def main(period_start: str, period_end: str, incident_info: str, alerts: str) -> dict:
    host         = os.environ.get("CH_HOST", "localhost")
    port         = int(os.environ.get("CH_PORT", "8123"))
    user         = os.environ.get("CH_USER", "default")
    password     = os.environ.get("CH_PASSWORD", "")
    database     = os.environ.get("CH_DATABASE", "default")
    batch_size   = int(os.environ.get("BATCH_SIZE", "200"))
    token_budget = int(os.environ.get("CHUNK_TOKEN_BUDGET", "6000"))

    all_rows = []
    last_ts = period_start

    while True:
        page_rows = []
        for sql_tmpl in [CONTAINERS_SQL, EVENTS_SQL]:
            sql = sql_tmpl.format(
                database=database,
                last_ts=last_ts,
                period_end=period_end,
                limit=batch_size,
            )
            page_rows.extend(ch_query(host, port, user, password, sql))

        if not page_rows:
            break

        page_rows.sort(key=lambda r: r.get("timestamp", ""))
        all_rows.extend(r["raw_line"] for r in page_rows)

        last_end = max(r.get("end_time", r.get("timestamp", "")) for r in page_rows)
        if last_end <= last_ts:
            break
        last_ts = last_end

        if len(page_rows) < batch_size:
            break

    batches = chunk_rows(all_rows, token_budget=token_budget)
    return {"batches": batches, "batch_count": len(batches)}
