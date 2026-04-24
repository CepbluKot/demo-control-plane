"""Dify Code Node: Load & Chunk

Copy-paste в Dify Code Node (Python).
Inputs:
  period_start      (str) — начало периода, ISO 8601
  period_end        (str) — конец периода, ISO 8601
  incident_info     (str) — описание инцидента
  alerts            (str) — тексты алертов
  ch_host           (str) — ClickHouse host
  ch_port           (str) — ClickHouse port (default "8123")
  ch_user           (str) — ClickHouse user
  ch_password       (str) — ClickHouse password
  ch_database       (str) — ClickHouse database (default "default")
  batch_size        (str) — строк на страницу пагинации (default "200")
  chunk_token_budget(str) — токенов на батч (default "6000")
  active_queries    (str) — индексы запросов из SQL_QUERIES через запятую (default "0,1")
Outputs: batches (Array[String]), batch_count (Number)
  Число батчей не ограничено — размер каждого задаётся rows_per_batch.

Требования к SQL-шаблонам в SQL_QUERIES:
  - плейсхолдеры: {database}, {last_ts}, {period_end}, {limit}
  - обязательные колонки: timestamp (DateTime), end_time (DateTime)
  - остальные поля — произвольные, попадут в JSON каждой строки батча
"""
import json
import urllib.request
from datetime import datetime


# ── ClickHouse ────────────────────────────────────────────────────────

def ch_query(host, port, user, password, sql, timeout=120):
    """POST-запрос к ClickHouse HTTP interface. Возвращает list[dict]."""
    url = f"http://{host}:{port}/"
    body = (sql + " FORMAT JSONEachRow").encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("X-ClickHouse-User", user)
    req.add_header("X-ClickHouse-Key", password)
    req.add_header("Content-Type", "text/plain; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            lines = resp.read().decode("utf-8").splitlines()
            return [json.loads(line) for line in lines if line.strip()]
    except urllib.request.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ClickHouse HTTP {e.code} at {url!r}: {body_text[:500]}") from e


# ── Chunker ───────────────────────────────────────────────────────────

_MAX_FIELD_CHARS = 2000  # обрезаем длинные поля чтобы один ряд не превысил лимит Dify


def _truncate(row):
    """Обрезает строковые поля до _MAX_FIELD_CHARS символов."""
    return {
        k: (v[:_MAX_FIELD_CHARS] + "…" if isinstance(v, str) and len(v) > _MAX_FIELD_CHARS else v)
        for k, v in row.items()
    }



def estimate_tokens(text):
    return max(1, len(text) // 4)


def chunk_rows(rows, token_budget=6000):
    """rows: list[dict]. Возвращает list[str] где каждый элемент — JSON-массив строк батча."""
    if not rows:
        return []
    chunks, current, current_tokens = [], [], 0
    for row in rows:
        row_str = json.dumps(row, ensure_ascii=False, default=str)
        t = estimate_tokens(row_str)
        if current and current_tokens + t > token_budget:
            chunks.append(json.dumps(current, ensure_ascii=False))
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += t
    if current:
        chunks.append(json.dumps(current, ensure_ascii=False))
    return chunks


# ── SQL Queries ───────────────────────────────────────────────────────
# Список SQL-шаблонов. Все должны возвращать: timestamp, end_time + любые поля.
# Чтобы добавить источник — допишите новый шаблон в список.
# active_queries выбирает по индексу (0-based).

SQL_QUERIES = [
    # 0: container logs — ошибки, сгруппированные по повторяющимся строкам
    """
    SELECT
        start_time AS timestamp,
        end_time,
        cnt,
        namespace,
        pod_name,
        container_name,
        log_text
    FROM (
        SELECT
            min(timestamp) AS start_time,
            max(timestamp) AS end_time,
            count() AS cnt,
            any(kubernetes_namespace_name) AS namespace,
            any(kubernetes_pod_name) AS pod_name,
            any(kubernetes_container_name) AS container_name,
            min(log) AS log_text
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
                FROM {database}.log_k8s_containers
                WHERE timestamp > parseDateTime64BestEffort('{last_ts}')
                  AND timestamp <= parseDateTime64BestEffort('{period_end}')
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
    ORDER BY timestamp ASC
    LIMIT {limit}
    """,

    # 1: k8s events
    """
    SELECT
        timestamp,
        timestamp AS end_time,
        reason,
        involvedObject_namespace AS namespace,
        involvedObject_name AS object_name,
        message
    FROM {database}.log_k8s_events
    WHERE timestamp > parseDateTime64BestEffort('{last_ts}')
      AND timestamp <= parseDateTime64BestEffort('{period_end}')
    ORDER BY timestamp ASC
    LIMIT {limit}
    """,
]


# ── Main ──────────────────────────────────────────────────────────────

def main(
    period_start: str,
    period_end: str,
    incident_info: str,
    alerts: str,
    ch_host: str = "localhost",
    ch_port: str = "8123",
    ch_user: str = "default",
    ch_password: str = "",
    ch_database: str = "default",
    batch_size: str = "200",
    active_queries: str = "0,1",
    rows_per_batch: str = "10",
) -> dict:
    start_dt = datetime.fromisoformat(period_start)
    end_dt   = datetime.fromisoformat(period_end)
    if (end_dt - start_dt).total_seconds() > 7 * 24 * 3600:
        raise ValueError(
            f"Период слишком большой: {(end_dt - start_dt).days} дн. "
            "Максимум — 7 дней. Сократи период."
        )

    host         = ch_host
    port         = int(ch_port)
    user         = ch_user
    password     = ch_password
    database     = ch_database
    batch_size   = int(batch_size)
    indices = [int(i.strip()) for i in active_queries.split(",") if i.strip()]
    sql_templates = []
    for idx in indices:
        if idx < 0 or idx >= len(SQL_QUERIES):
            raise ValueError(f"Индекс запроса {idx} выходит за границы (всего {len(SQL_QUERIES)})")
        sql_templates.append(SQL_QUERIES[idx])

    chunk_size = max(1, int(rows_per_batch))

    all_rows = []
    last_ts = period_start

    while True:
        page_rows = []
        for sql_tmpl in sql_templates:
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
        all_rows.extend(_truncate(r) for r in page_rows)

        last_end = max(r.get("end_time", r.get("timestamp", "")) for r in page_rows)
        if last_end <= last_ts:
            break
        last_ts = last_end

        if len(page_rows) < batch_size:
            break

    batches = [
        json.dumps(all_rows[i:i + chunk_size], ensure_ascii=False, default=str)
        for i in range(0, len(all_rows), chunk_size)
    ]
    return {"batches": batches, "batch_count": len(batches)}
