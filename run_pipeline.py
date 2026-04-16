"""
run_pipeline.py — запуск пайплайна log_summarizer для анализа инцидентов.

Режимы (переменная MODE):
  "incidents" — обрабатывает список INCIDENTS; время берётся из incident_start/end.
  "freeform"  — один инцидент с явным периодом FREEFORM_START..FREEFORM_END
                и алертами FREEFORM_ALERTS; удобно когда нужен произвольный отрезок.

Запуск:
    python run_pipeline.py
    python run_pipeline.py --only airflow-oom   # только один инцидент (только incidents)
    python run_pipeline.py --list               # показать список инцидентов
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Московское время — UTC+3, без летнего перевода.
# Используй MSK при задании временных окон инцидентов.
MSK = timezone(timedelta(hours=3))

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
    namespace,
    container_name,
    concat(
        '[', toString(start_time),
        ' → ', toString(end_time), ']',
        ' ×', toString(cnt),
        '  ', log_text
    )                                 AS raw_line
FROM (
    SELECT
        min(timestamp)                     AS start_time,   -- первое появление группы
        max(timestamp)                     AS end_time,     -- последнее появление группы
        min(log)                           AS log_text,     -- текст лога (одинаковый внутри группы)
        count()                            AS cnt,          -- сколько раз повторился подряд
        any(kubernetes_namespace_name)     AS namespace,
        any(kubernetes_container_name)     AS container_name
    FROM (
        SELECT *,
            sum(is_new_group) OVER (
                PARTITION BY kubernetes_namespace_name, kubernetes_container_name
                ORDER BY timestamp ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS group_id
        FROM (
            SELECT *,
                -- Новая группа = последние 10 символов лога изменились
                -- (группируем внутри одного контейнера, чтобы не мешать логи разных сервисов)
                if(
                    right(log, 10) != ifNull(
                        lagInFrame(right(log, 10)) OVER (
                            PARTITION BY kubernetes_namespace_name, kubernetes_container_name
                            ORDER BY timestamp ASC
                        ), ''
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
    GROUP BY group_id, kubernetes_namespace_name, kubernetes_container_name
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
        # fired_at — в МСК (как зафиксировано в мониторинге).
        "alerts": [
            {
                "name": "AirflowKubernetesExecutorFailed",
                "fired_at": datetime(2026, 3, 18, 5, 8, 0, tzinfo=MSK),
                "severity": "critical",
                "description": "Pod creation failed (Forbidden) in kubernetes_executor",
            },
            {
                "name": "AirflowDAGRunStuck",
                "fired_at": datetime(2026, 3, 18, 5, 15, 0, tzinfo=MSK),
                "severity": "high",
            },
        ],
        # Узкое окно: когда наблюдались проблемы и сработали алерты (МСК).
        # Алерты должны попасть ВНУТРЬ этого окна.
        "incident_start": datetime(2026, 3, 18, 5, 5, 0, tzinfo=MSK),
        "incident_end":   datetime(2026, 3, 18, 5, 45, 0, tzinfo=MSK),

        # Широкое окно: откуда грузить логи (контекст вокруг инцидента, МСК).
        # По умолчанию ±1 час от incident window. Можно убрать — тогда == incident.
        "context_start": datetime(2026, 3, 18, 4, 0, 0, tzinfo=MSK),
        "context_end":   datetime(2026, 3, 18, 7, 0, 0, tzinfo=MSK),
    },
    # Пример второго инцидента — раскомментируй и заполни:
    # {
    #     "name": "airflow-oom-2026-03-20",
    #     "context": """
    #         Airflow scheduler упал с OOM. Задачи не планируются ~20 минут.
    #         Алерт: AirflowSchedulerNotRunning, 17:32 МСК.
    #     """,
    #     "incident_start": datetime(2026, 3, 20, 17, 20, 0, tzinfo=MSK),
    #     "incident_end":   datetime(2026, 3, 20, 18, 30, 0, tzinfo=MSK),
    # },
]

# ══════════════════════════════════════════════════════════════════════
#  РЕЖИМ ЗАПУСКА
#
#  "incidents" — обрабатывает список INCIDENTS ниже (дефолт).
#               Время инцидента и контекстное окно берутся из каждой записи.
#
#  "freeform"  — один инцидент с явным отрезком FREEFORM_START..FREEFORM_END.
#               Алерты и контекст задаются через FREEFORM_ALERTS / FREEFORM_CONTEXT.
#               Удобно когда нужно разобрать произвольный период без добавления
#               записи в INCIDENTS.
# ══════════════════════════════════════════════════════════════════════

MODE = "incidents"   # "incidents" | "freeform"

# ══════════════════════════════════════════════════════════════════════
#  AUTO_TRIM_AFTER_LAST_ALERT
#
#  True  — конец периода загрузки логов обрезается до
#          max(alert.fired_at) + TRIM_BUFFER_MINUTES.
#          Логи после последнего алерта обычно не нужны для анализа причины.
#  False — конец периода не меняется (context_end / FREEFORM_END как задано).
#
#  Работает в обоих режимах: incidents и freeform.
# ══════════════════════════════════════════════════════════════════════

AUTO_TRIM_AFTER_LAST_ALERT = False
TRIM_BUFFER_MINUTES        = 15   # сколько минут оставить после последнего алерта

# ══════════════════════════════════════════════════════════════════════
#  FREEFORM MODE — заполни если MODE = "freeform"
#
#  FREEFORM_START / FREEFORM_END — явный период загрузки логов (МСК).
#  Алерты должны попасть внутрь этого диапазона.
#  FREEFORM_CONTEXT — описание для LLM: что случилось, что ищем.
# ══════════════════════════════════════════════════════════════════════

FREEFORM_START:   datetime | None = None   # datetime(2026, 3, 18, 1, 30, 0, tzinfo=MSK)
FREEFORM_END:     datetime | None = None   # datetime(2026, 3, 18, 19, 30, 0, tzinfo=MSK)
FREEFORM_CONTEXT: str             = ""     # "Разобраться что происходило с кластером"
FREEFORM_ALERTS: list[dict]       = [
    # Тот же формат что в INCIDENTS:
    # {"name": "AlertName", "fired_at": datetime(..., tzinfo=MSK), "severity": "critical", "description": "..."},
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

# Максимум токенов на один MAP-батч (размер одного чанка логов в промпте).
# None → автоматически 55% от MAX_CONTEXT_TOKENS.
# Задай явно если нужен контроль: например 40_000 при слабой модели,
# или 80_000 если хочешь меньше батчей при большом контексте.
MAX_BATCH_TOKENS = None

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


async def run_incident(ch, incident: dict, run_dir: Path) -> tuple[str, str | Exception, object]:
    """Запускает пайплайн для одного инцидента.

    Args:
        run_dir: Корневая папка текущего запуска (runs/{run_timestamp}/).
                 Артефакты инцидента попадут в run_dir/{name}/{artifact_timestamp}/.

    Returns:
        (name, report | error, orchestrator | None)
        orchestrator.last_merged  — MergedAnalysis после успешного прогона.
        orchestrator.run_dir      — путь к папке с артефактами (для сохранения report.md).
    """
    from log_summarizer.config import PipelineConfig
    from log_summarizer.models import make_alerts
    from log_summarizer.orchestrator import PipelineOrchestrator

    name = incident["name"]
    log  = logging.getLogger(f"run_pipeline.{name}")

    inc_start = incident.get("incident_start") or incident.get("start")
    inc_end   = incident.get("incident_end")   or incident.get("end")
    log.info("=== START: %s  [%s → %s] ===", name, inc_start, inc_end)
    ctx_start = incident.get("context_start")
    ctx_end   = incident.get("context_end")

    config = PipelineConfig(
        logs_sql_template=LOGS_SQL.strip(),
        metrics_sql_template=METRICS_SQL.strip() or None,
        incident_context=incident["context"].strip(),
        incident_start=inc_start,
        incident_end=inc_end,
        context_start=ctx_start,
        context_end=ctx_end,
        alerts=make_alerts(incident.get("alerts", [])),
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        model_supports_tool_calling=MODEL_SUPPORTS_TOOL_CALLING,
        map_concurrency=MAP_CONCURRENCY,
        batch_size=BATCH_SIZE,
        max_events_per_merge=MAX_EVENTS_PER_MERGE,
        max_batch_tokens=MAX_BATCH_TOKENS,
        # Артефакты: runs/{run_timestamp}/{incident_name}/
        # Оркестратор добавит ещё один уровень {artifact_timestamp}/ внутри.
        runs_dir=str(run_dir / name),
    )

    orchestrator = PipelineOrchestrator(ch, config)
    try:
        report = await orchestrator.run()
        log.info("=== DONE: %s ===", name)
        return name, report, orchestrator
    except Exception as exc:
        log.error("=== FAILED: %s — %s ===", name, exc)
        return name, exc, None


def _apply_trim(incident: dict, log: logging.Logger) -> dict:
    """Обрезает context_end до max(alert.fired_at) + TRIM_BUFFER_MINUTES.

    Возвращает изменённую копию словаря инцидента.
    Если алертов нет или ни у одного нет fired_at — возвращает оригинал без изменений.
    """
    alerts = incident.get("alerts", [])
    fired_times = [a["fired_at"] for a in alerts if a.get("fired_at")]
    if not fired_times:
        return incident

    last_alert = max(fired_times)
    trim_end = last_alert + timedelta(minutes=TRIM_BUFFER_MINUTES)

    ctx_end = incident.get("context_end") or incident.get("incident_end")
    if ctx_end and trim_end >= ctx_end:
        # Обрезать нечего — конец и так раньше или совпадает
        return incident

    log.info(
        "AUTO_TRIM: context_end %s → %s (последний алерт %s + %d мин)",
        ctx_end, trim_end, last_alert, TRIM_BUFFER_MINUTES,
    )
    result = dict(incident)
    result["context_end"] = trim_end
    return result


async def main(only: str | None) -> None:
    from log_summarizer.utils.logging import setup_pipeline_logging

    # ── Корневая папка этого запуска ──────────────────────────────────
    run_ts = datetime.now(MSK).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path(RUNS_DIR) / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "pipeline.log" if LOG_FILE else None
    setup_pipeline_logging(LOG_LEVEL, log_file=str(log_path) if log_path else None)

    log = logging.getLogger("run_pipeline")
    log.info("Папка запуска: %s", run_dir.resolve())

    try:
        import clickhouse_connect
    except ImportError:
        raise SystemExit("clickhouse-connect не установлен: pip install clickhouse-connect")

    ch = clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        username=CH_USER, password=CH_PASSWORD,
        database=CH_DATABASE,
    )

    # ── Freeform mode ─────────────────────────────────────────────────
    if MODE == "freeform":
        if not FREEFORM_START or not FREEFORM_END:
            raise SystemExit(
                "FREEFORM_START и FREEFORM_END должны быть заданы для MODE=\"freeform\".\n"
                "Заполни их в run_pipeline.py."
            )
        if not FREEFORM_ALERTS:
            raise SystemExit(
                "FREEFORM_ALERTS пуст. Укажи хотя бы один алерт — "
                "без алертов анализ инцидента не имеет смысла."
            )
        # Проверяем что все алерты попадают в заданный период
        for a in FREEFORM_ALERTS:
            ft = a.get("fired_at")
            if ft and not (FREEFORM_START <= ft <= FREEFORM_END):
                raise SystemExit(
                    f"Алерт '{a['name']}' fired_at={ft} выходит за пределы "
                    f"FREEFORM_START..FREEFORM_END ({FREEFORM_START} — {FREEFORM_END})."
                )
        name = "freeform-" + FREEFORM_START.strftime("%Y-%m-%dT%H-%M")
        incident = {
            "name": name,
            "context": FREEFORM_CONTEXT or f"Анализ логов за период {FREEFORM_START} — {FREEFORM_END}",
            "alerts": FREEFORM_ALERTS,
            "incident_start": FREEFORM_START,
            "incident_end":   FREEFORM_END,
            "context_start":  FREEFORM_START,
            "context_end":    FREEFORM_END,
        }
        if AUTO_TRIM_AFTER_LAST_ALERT:
            incident = _apply_trim(incident, log)
        log.info("Freeform mode: %s → %s", FREEFORM_START, FREEFORM_END)
        name, result, orchestrator = await run_incident(ch, incident, run_dir)
        if isinstance(result, Exception):
            raise SystemExit(f"Пайплайн завершился с ошибкой: {result}")
        artifact_dir = orchestrator.run_dir if orchestrator is not None else None
        if artifact_dir is not None:
            mono_path = artifact_dir / "report.md"
            mono_path.write_text(result, encoding="utf-8")
            log.info("report.md → %s", mono_path.resolve())
        log.info("Freeform mode завершён. Артефакты: %s", run_dir.resolve())
        return

    # ── Incidents mode ────────────────────────────────────────────────
    incidents = INCIDENTS
    if only:
        incidents = [i for i in INCIDENTS if i["name"] == only]
        if not incidents:
            raise SystemExit(
                f"Инцидент '{only}' не найден. "
                f"Доступные: {[i['name'] for i in INCIDENTS]}"
            )

    failed = []
    # (name, MergedAnalysis, PipelineConfig) — для комбинированного отчёта
    merged_list: list[tuple[str, object, object]] = []

    for incident in incidents:
        if AUTO_TRIM_AFTER_LAST_ALERT:
            incident = _apply_trim(incident, log)
        name, result, orchestrator = await run_incident(ch, incident, run_dir)
        if isinstance(result, Exception):
            failed.append(name)
            continue

        # Монолитный отчёт кладём рядом с report_multipass.md
        # (в ту же папку {artifact_timestamp}/, которую создал оркестратор)
        artifact_dir = orchestrator.run_dir if orchestrator is not None else None
        if artifact_dir is not None:
            mono_path = artifact_dir / "report.md"
            mono_path.write_text(result, encoding="utf-8")
            log.info("report.md (monolithic) → %s", mono_path.resolve())
        else:
            # fallback: рядом с папкой инцидента
            fallback = run_dir / name / "report.md"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(result, encoding="utf-8")
            log.info("report.md (monolithic, fallback) → %s", fallback.resolve())

        if orchestrator is not None and orchestrator.last_merged is not None:
            merged_list.append((name, orchestrator.last_merged, orchestrator.config))

    total = len(incidents)
    log.info("")
    log.info(
        "══ ИТОГ: %d/%d инцидентов обработано%s",
        total - len(failed), total,
        f"  ·  ошибки: {failed}" if failed else "",
    )
    for name, merged, cfg in merged_list:
        log.info(
            "  ✓ %-40s  %d событий  ·  %d гипотез  ·  %d строк логов",
            name,
            len(merged.events),        # type: ignore[attr-defined]
            len(merged.hypotheses),    # type: ignore[attr-defined]
            getattr(cfg, "total_log_rows", 0),
        )
    log.info("  Все артефакты этого запуска: %s", run_dir.resolve())

    # ── Комбинированный отчёт (если ≥2 инцидентов) ───────────────────
    if len(merged_list) >= 2:
        log.info("")
        log.info(
            "Запуск комбинированного отчёта (%d инцидентов → combined_report.md)...",
            len(merged_list),
        )
        from log_summarizer.cross_incident_analyzer import CrossIncidentAnalyzer
        from log_summarizer.llm_client import LLMClient

        cross_llm = LLMClient(
            api_base=API_BASE,
            api_key=API_KEY,
            model=MODEL,
            max_retries=3,
            retry_backoff_base=2.0,
            use_instructor=True,
            model_supports_tool_calling=MODEL_SUPPORTS_TOOL_CALLING,
            audit_dir=None,
        )
        # combined_report.md кладём прямо в папку запуска
        analyzer = CrossIncidentAnalyzer(cross_llm, runs_dir=str(run_dir))
        try:
            await analyzer.generate_combined_report(merged_list)
        except Exception as exc:
            log.error("Комбинированный отчёт завершился с ошибкой: %s", exc)

    elif len(merged_list) == 1:
        log.info(
            "Один инцидент — комбинированный отчёт не нужен. "
            "Отчёт: %s",
            run_dir / merged_list[0][0],
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, metavar="NAME",
                   help="Запустить только один инцидент по имени (только режим incidents)")
    p.add_argument("--list", action="store_true",
                   help="Показать список инцидентов и выйти")
    args = p.parse_args()

    if args.list:
        if MODE == "freeform":
            print(f"  freeform  {FREEFORM_START} → {FREEFORM_END}")
        else:
            for i in INCIDENTS:
                s = i.get("incident_start") or i.get("start", "?")
                e = i.get("incident_end")   or i.get("end",   "?")
                print(f"  {i['name']}  {s} → {e}")
    else:
        asyncio.run(main(args.only))
