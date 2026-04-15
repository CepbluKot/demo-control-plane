"""
run_pipeline.py — запуск пайплайна log_summarizer для списка инцидентов.

Заполни INCIDENTS и запусти:
    python run_pipeline.py
    python run_pipeline.py --only airflow-oom  # один инцидент по имени
    python run_pipeline.py --list              # показать список инцидентов
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  LLM
# ══════════════════════════════════════════════════════════════════════

# Адрес vLLM / OpenAI-совместимого сервера (без /v1 — добавится автоматически)
API_BASE = "http://localhost:8000"

# API-ключ (для vLLM обычно любая строка)
API_KEY = "sk-placeholder"

# Название модели — точно как в /v1/models или как её называет vLLM
MODEL = "PNX.QWEN3 235b a22b instruct"

# Размер контекстного окна модели в токенах
# Пайплайн использует ~55% под логи, остальное — под промпты и ответ
MAX_CONTEXT_TOKENS = 100_000

# False  — JSON mode через instructor (безопасно, работает с любым vLLM)
# True   — TOOLS mode (быстрее, но vLLM должен поддерживать tool calling)
MODEL_SUPPORTS_TOOL_CALLING = False

# ══════════════════════════════════════════════════════════════════════
#  ClickHouse
# ══════════════════════════════════════════════════════════════════════

CH_HOST     = "localhost"   # хост или IP ClickHouse
CH_PORT     = 8123          # HTTP-порт (8123 по умолчанию)
CH_USER     = "default"
CH_PASSWORD = ""
CH_DATABASE = "default"     # база данных по умолчанию (можно переопределить в SQL)

# ══════════════════════════════════════════════════════════════════════
#  SQL-шаблон для логов (общий для всех инцидентов)
#
#  Плейсхолдеры — DataLoader подставляет их автоматически:
#    {start_time}  — начало периода инцидента (ISO8601)
#    {end_time}    — конец периода инцидента (ISO8601)
#    {last_ts}     — keyset-пагинация: max timestamp предыдущей страницы;
#                    первый вызов = {start_time}; заменяет OFFSET
#    {limit}       — количество строк на страницу (= BATCH_SIZE ниже)
#
#  Обязательные колонки в SELECT:
#    timestamp     — DataLoader ищет именно это имя для keyset и сортировки
#    raw_line      — текст лога, передаётся в LLM дословно; если колонки нет —
#                    DataLoader попробует: message / msg / log / value
#
#  Не используй OFFSET — на больших таблицах падает по памяти.
#  Вместо этого фильтруй по start_time > '{last_ts}' (см. WHERE внизу).
# ══════════════════════════════════════════════════════════════════════
LOGS_SQL = """
SELECT
    start_time                        AS timestamp,
    concat(
        '[', toString(start_time),
        ' → ', toString(end_time), ']',
        ' ×', toString(cnt),
        '  ', log_text
    )                                 AS raw_line
FROM (
    SELECT
        min(timestamp) AS start_time,   -- первое появление группы
        max(timestamp) AS end_time,     -- последнее появление группы
        min(log)       AS log_text,     -- текст лога (одинаковый внутри группы)
        count()        AS cnt           -- сколько раз повторился подряд
    FROM (
        SELECT *,
            sum(is_new_group) OVER (
                ORDER BY timestamp ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS group_id
        FROM (
            SELECT *,
                -- Новая группа = последние 10 символов лога изменились
                if(
                    right(log, 10) != ifNull(
                        lagInFrame(right(log, 10)) OVER (ORDER BY timestamp ASC), ''
                    ),
                    1, 0
                ) AS is_new_group
            FROM raw_lm.log_k8s_containers_MT   -- ← таблица с логами
            WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
              AND ext_ClusterName = 'ndp-p01'                     -- ← кластер
              AND kubernetes_container_name LIKE '%airflow%'       -- ← фильтр контейнеров
              AND (   kubernetes_namespace_name LIKE '%airflow%'   -- ← namespace
                   OR kubernetes_namespace_name LIKE '%kube%')
              -- Оставляем только строки с признаками ошибок
              AND multiSearchAny(lower(log), [
                    'fatal', 'critical', 'error', 'exception',
                    'alert', 'panic', 'failed', 'failure', 'crash', 'abort',
                    'timeout', 'timed out', 'deadlock',
                    'out of memory', 'oom', 'disk full',
                    'no space left', 'permission denied',
                    'access denied', 'unauthorized', 'forbidden',
                    'connection refused', 'connection reset',
                    'ssl error', 'segfault', 'killed',
                    'rollback', 'traceback', 'stack trace'
                ])
              -- Исключаем шумные строки которые ложно попадают под фильтр выше
              AND NOT multiSearchAny(lower(log), [
                    'certificate_verify_failed',
                    'info',
                    'object has no attribute ''upper'''
                ])
        )
    )
    GROUP BY group_id
)
-- Keyset-пагинация: берём только строки новее последнего полученного timestamp.
-- DataLoader обновляет {last_ts} после каждой страницы автоматически.
WHERE start_time > '{last_ts}'
ORDER BY start_time ASC
LIMIT {limit}
"""

# SQL для метрик — опционально.
# Те же плейсхолдеры: {start_time}, {end_time}.
# Обязательные колонки: timestamp, metric_name, value, service (или source).
# Оставь пустым если метрики не нужны.
METRICS_SQL = ""

# ══════════════════════════════════════════════════════════════════════
#  INCIDENTS — список инцидентов для анализа
#
#  name     — уникальный slug; используется как имя папки в runs/ и файла отчёта.
#             Только латиница, цифры, дефисы (без пробелов).
#
#  context  — свободный текст для LLM: что случилось, какие сервисы затронуты,
#             какие алерты сработали. Чем конкретнее — тем точнее анализ.
#             Можно вставить текст алерта, описание из тикета и т.п.
#
#  start    — начало периода для выборки логов. Бери с запасом: за 10-15 минут
#             до первого алерта, чтобы LLM видел предысторию.
#             Обязательно указывай timezone (tzinfo=timezone.utc или +3 и т.д.)
#
#  end      — конец периода. Бери до момента полного восстановления + запас.
# ══════════════════════════════════════════════════════════════════════
INCIDENTS = [
    {
        "name": "airflow-forbidden-pods-2026-03-18",
        "context": """
            Airflow: массовые ошибки Pod creation failed (Forbidden) в kubernetes_executor.
            DAG-раны зависают в state=running, воркеры убиваются SIGTERM.
        """,
        # Алерты: name обязателен; fired_at, severity, description — опционально.
        # ID присваиваются автоматически: alert-001, alert-002, ...
        "alerts": [
            {
                "name": "AirflowKubernetesExecutorFailed",
                "fired_at": datetime(2026, 3, 18, 2, 8, 0, tzinfo=timezone.utc),
                "severity": "critical",
                "description": "Pod creation failed (Forbidden) in kubernetes_executor",
            },
            {
                "name": "AirflowDAGRunStuck",
                "fired_at": datetime(2026, 3, 18, 2, 15, 0, tzinfo=timezone.utc),
                "severity": "high",
            },
        ],
        "start": datetime(2026, 3, 18, 2, 0, 0, tzinfo=timezone.utc),
        "end":   datetime(2026, 3, 18, 3, 0, 0, tzinfo=timezone.utc),
    },
    # Пример второго инцидента — раскомментируй и заполни:
    # {
    #     "name": "airflow-oom-2026-03-20",
    #     "context": """
    #         Airflow scheduler упал с OOM. Задачи не планируются ~20 минут.
    #         Алерт: AirflowSchedulerNotRunning, 14:32 UTC.
    #     """,
    #     "start": datetime(2026, 3, 20, 14, 20, 0, tzinfo=timezone.utc),
    #     "end":   datetime(2026, 3, 20, 15, 30, 0, tzinfo=timezone.utc),
    # },
]

# ══════════════════════════════════════════════════════════════════════
#  Параметры пайплайна
# ══════════════════════════════════════════════════════════════════════

# Сколько чанков логов обрабатывать параллельно (параллельные вызовы LLM).
# Больше = быстрее, но выше нагрузка на LLM-сервер.
MAP_CONCURRENCY = 5

# Сколько строк тащить из ClickHouse за один запрос.
# Не влияет на размер промпта — после выборки строки нарезаются по токенам.
BATCH_SIZE = 1000

# Максимум событий (events) в MergedAnalysis после каждого REDUCE-шага.
# При превышении самые незначительные (по severity) отбрасываются.
MAX_EVENTS_PER_MERGE = 30

# Папка для артефактов: runs/{incident_name}/llm/ map/ reduce/ report.md
RUNS_DIR = "runs"

# Файл для логов пайплайна. None — только stderr.
LOG_FILE = "pipeline.log"

# INFO  — старт/стоп каждого шага, пути к сохранённым файлам, предупреждения
# DEBUG — детали: токены, размеры батчей, страницы ClickHouse
LOG_LEVEL = "INFO"

# ══════════════════════════════════════════════════════════════════════


async def run_incident(ch, incident: dict) -> tuple[str, str | Exception]:
    """Запускает пайплайн для одного инцидента. Возвращает (name, report | error)."""
    from log_summarizer.config import PipelineConfig
    from log_summarizer.models import make_alerts
    from log_summarizer.orchestrator import PipelineOrchestrator

    name = incident["name"]
    log  = logging.getLogger(f"run_pipeline.{name}")
    log.info("=== START: %s  [%s → %s] ===", name, incident["start"], incident["end"])

    config = PipelineConfig(
        logs_sql_template=LOGS_SQL.strip(),
        metrics_sql_template=METRICS_SQL.strip() or None,
        incident_context=incident["context"].strip(),
        incident_start=incident["start"],
        incident_end=incident["end"],
        alerts=make_alerts(incident.get("alerts", [])),
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        model_supports_tool_calling=MODEL_SUPPORTS_TOOL_CALLING,
        map_concurrency=MAP_CONCURRENCY,
        batch_size=BATCH_SIZE,
        max_events_per_merge=MAX_EVENTS_PER_MERGE,
        runs_dir=str(Path(RUNS_DIR) / name),
    )

    try:
        report = await PipelineOrchestrator(ch, config).run()
        log.info("=== DONE: %s ===", name)
        return name, report
    except Exception as exc:
        log.error("=== FAILED: %s — %s ===", name, exc)
        return name, exc


async def main(only: str | None) -> None:
    from log_summarizer.utils.logging import setup_pipeline_logging
    setup_pipeline_logging(LOG_LEVEL, log_file=LOG_FILE or None)

    log = logging.getLogger("run_pipeline")

    try:
        import clickhouse_connect
    except ImportError:
        raise SystemExit("clickhouse-connect не установлен: pip install clickhouse-connect")

    ch = clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        username=CH_USER, password=CH_PASSWORD,
        database=CH_DATABASE,
    )

    incidents = INCIDENTS
    if only:
        incidents = [i for i in INCIDENTS if i["name"] == only]
        if not incidents:
            raise SystemExit(
                f"Инцидент '{only}' не найден. "
                f"Доступные: {[i['name'] for i in INCIDENTS]}"
            )

    Path(RUNS_DIR).mkdir(exist_ok=True)

    failed = []
    for incident in incidents:
        name, result = await run_incident(ch, incident)
        if isinstance(result, Exception):
            failed.append(name)
            continue

        report_path = Path(RUNS_DIR) / incident["name"] / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(result, encoding="utf-8")
        log.info("Report → %s", report_path.resolve())

    total = len(incidents)
    log.info(
        "Done: %d/%d инцидентов обработано%s",
        total - len(failed), total,
        f", ошибки: {failed}" if failed else "",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, metavar="NAME",
                   help="Запустить только один инцидент по имени")
    p.add_argument("--list", action="store_true",
                   help="Показать список инцидентов и выйти")
    args = p.parse_args()

    if args.list:
        for i in INCIDENTS:
            print(f"  {i['name']}  {i['start']} → {i['end']}")
    else:
        asyncio.run(main(args.only))
