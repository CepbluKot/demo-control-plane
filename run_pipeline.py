"""
run_pipeline.py — запуск пайплайна log_summarizer для анализа инцидентов.

Режимы (переменная MODE):
  "incidents" — обрабатывает список INCIDENTS; время берётся из incident_start/end.
  "freeform"  — один инцидент с явным периодом FREEFORM_START..FREEFORM_END
                и алертами FREEFORM_ALERTS; удобно когда нужен произвольный отрезок.
  "context"   — как freeform, но без обязательных алертов: есть только окно логов
                и текстовый комментарий/контекст SRE.

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

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Московское время — UTC+3, без летнего перевода.
# Используй MSK при задании временных окон инцидентов.
MSK = timezone(timedelta(hours=3))


class RuntimeSettings(BaseSettings):
    """Runtime-настройки из окружения и .env.

    Значения ниже остаются дефолтами для локального запуска, но в реальном
    контуре их можно переопределять через .env или environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_base: str = Field(
        "http://localhost:8000",
        validation_alias=AliasChoices("API_BASE", "LLM_API_BASE"),
    )
    api_key: str = Field(
        "sk-placeholder",
        validation_alias=AliasChoices("API_KEY", "LLM_API_KEY"),
    )
    model: str = Field(
        "PNX.QWEN3 235b a22b instruct",
        validation_alias=AliasChoices("MODEL", "LLM_MODEL"),
    )
    max_context_tokens: int = Field(
        100_000,
        validation_alias=AliasChoices("MAX_CONTEXT_TOKENS", "LLM_CONTEXT_TOKENS"),
    )
    model_supports_tool_calling: bool = Field(
        False,
        validation_alias=AliasChoices("MODEL_SUPPORTS_TOOL_CALLING", "LLM_TOOL_CALLING"),
    )

    ch_host: str = Field("localhost", validation_alias="CH_HOST")
    ch_port: int = Field(8123, validation_alias="CH_PORT")
    ch_user: str = Field("default", validation_alias="CH_USER")
    ch_password: str = Field("", validation_alias="CH_PASSWORD")
    ch_database: str = Field("default", validation_alias="CH_DATABASE")


SETTINGS = RuntimeSettings()

# ══════════════════════════════════════════════════════════════════════
#  LLM
# ══════════════════════════════════════════════════════════════════════

# Адрес vLLM / OpenAI-совместимого сервера (без /v1 — добавится автоматически)
API_BASE = SETTINGS.api_base

# API-ключ (для vLLM обычно любая строка)
API_KEY = SETTINGS.api_key

# Название модели — точно как в /v1/models или как её называет vLLM
MODEL = SETTINGS.model

# Размер контекстного окна модели в токенах
# Пайплайн использует ~55% под логи, остальное — под промпты и ответ
MAX_CONTEXT_TOKENS = SETTINGS.max_context_tokens

# False  — JSON mode через instructor (безопасно, работает с любым vLLM)
# True   — TOOLS mode (быстрее, но vLLM должен поддерживать tool calling)
MODEL_SUPPORTS_TOOL_CALLING = SETTINGS.model_supports_tool_calling

# ══════════════════════════════════════════════════════════════════════
#  ClickHouse
# ══════════════════════════════════════════════════════════════════════

CH_HOST     = SETTINGS.ch_host       # хост или IP ClickHouse
CH_PORT     = SETTINGS.ch_port       # HTTP-порт (8123 по умолчанию)
CH_USER     = SETTINGS.ch_user
CH_PASSWORD = SETTINGS.ch_password
CH_DATABASE = SETTINGS.ch_database   # база данных по умолчанию (можно переопределить в SQL)

# ══════════════════════════════════════════════════════════════════════
#  LOGS_SQLS — список SQL-шаблонов для загрузки логов
#
#  Каждый элемент — отдельный источник данных. DataLoader запрашивает
#  их независимо с одним last_ts-watermark и сортирует результаты в Python.
#
#  Обязательные плейсхолдеры (проверяются при старте):
#    {last_ts}     — keyset-watermark; фильтрует timestamp > last_ts
#    {period_end}  — верхняя граница периода; фильтрует timestamp <= period_end
#    {limit}       — LIMIT на внешнем GROUP BY (= BATCH_SIZE групп на страницу)
#    {raw_limit}   — LIMIT на внутренних сырых строках (= BATCH_SIZE × BATCH_RAW_MULTIPLIER)
#
#  Обязательные колонки в SELECT:
#    timestamp     — start_time группы; используется для сортировки и zone-разметки
#    end_time      — max timestamp сырых строк группы; используется для watermark
#    raw_line      — текст для LLM; формат: [start → end] ×N  ns/pod  <текст>
#
#  Не используй OFFSET — на больших таблицах OOM.
#  Тег [EVT:reason] в raw_line помогает LLM отличать K8s-события от логов.
# ══════════════════════════════════════════════════════════════════════

LOGS_SQLS = [
    # ── Источник 1: Spark driver logs (k8s_logs.k-ndp-p11-ndp-flex-spark) ───
    # Схлопывает подряд идущие одинаковые message внутри cluster/pod/container.
    # namespace здесь заполняется значением cluster, потому что в таблице нет namespace.
    """
SELECT
    start_time                AS timestamp,
    end_time,
    namespace,
    pod_name,
    container_name,
    node_name,
    concat(
        '[', toString(start_time),
        ' → ', toString(end_time), ']',
        ' ×', toString(cnt),
        '  ', namespace, '/', pod_name,
        '  [LOG:', container_name, '@', node_name, '] ', message_text
    )                         AS raw_line
FROM (
    SELECT
        min(timestamp)         AS start_time,
        formatDateTime(max(timestamp), '%Y-%m-%d %H:%i:%S.%f', 'Europe/Moscow') AS end_time,
        any(message)           AS message_text,
        count()                AS cnt,
        any(cluster)           AS namespace,
        any(pod)               AS pod_name,
        any(container)         AS container_name,
        any(node)              AS node_name
    FROM (
        SELECT *,
            sum(is_new_group) OVER (
                PARTITION BY cluster, pod, container
                ORDER BY timestamp ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS group_id
        FROM (
            SELECT *,
                if(
                    message != ifNull(
                        lagInFrame(message) OVER (
                            PARTITION BY cluster, pod, container
                            ORDER BY timestamp ASC
                        ), ''
                    ),
                    1, 0
                ) AS is_new_group
            FROM k8s_logs.`k-ndp-p11-ndp-flex-spark`
            WHERE timestamp >  parseDateTime64BestEffort('{last_ts}', 6, 'Europe/Moscow')
              AND timestamp <= parseDateTime64BestEffort('{period_end}', 6, 'Europe/Moscow')
              AND cluster = 'ndp-p11'
              AND pod = 'fl32cfad0d758f29068a909050374e0d-65f0849eee1af10c-driver'
            ORDER BY timestamp ASC
            LIMIT {raw_limit}
        )
    )
    GROUP BY group_id, cluster, pod, container
)
ORDER BY start_time ASC
LIMIT {limit}
""",
]

# ── Валидация SQL-шаблонов ────────────────────────────────────────────────────
# Проверяется один раз при старте main(). SystemExit если что-то не так.

_REQUIRED_PLACEHOLDERS = ["{last_ts}", "{period_end}", "{limit}", "{raw_limit}"]
_REQUIRED_COLUMNS      = ["timestamp", "end_time", "raw_line"]

def _validate_sql_templates(sqls: list[str]) -> None:
    """Проверяет что все SQL-шаблоны содержат нужные плейсхолдеры и колонки."""
    if not sqls:
        raise SystemExit("LOGS_SQLS пуст — добавь хотя бы один SQL-шаблон.")
    errors = []
    for i, sql in enumerate(sqls):
        label = f"LOGS_SQLS[{i}]"
        missing_ph = [p for p in _REQUIRED_PLACEHOLDERS if p not in sql]
        if missing_ph:
            errors.append(f"{label}: отсутствуют плейсхолдеры {missing_ph}")
        missing_col = [c for c in _REQUIRED_COLUMNS if c not in sql]
        if missing_col:
            errors.append(f"{label}: отсутствуют колонки {missing_col} в SELECT")
    if errors:
        raise SystemExit("Ошибки в LOGS_SQLS:\n" + "\n".join(f"  • {e}" for e in errors))

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
        "name": "spark-driver-pod-finished-no-kube-logs",
        "context": """
            fl32cfad0d758f29068a909050374e0d-65f0849eee1af10c-driver pod завершает работу,
            при этом в Kubernetes логов pod не обнаруживается.
            Такие кейсы не единичны, происходят периодически, задания могут падать абсолютно разные.
        """,
        # Алерты: name обязателен; fired_at, severity, description — опционально.
        # ID присваиваются автоматически: alert-001, alert-002, ...
        # fired_at — в МСК (как зафиксировано в мониторинге).
        "alerts": [
            {
                "name": "SparkDriverPodFinishedWithoutKubeLogs",
                "fired_at": datetime(2026, 6, 22, 10, 2, 0, tzinfo=MSK),
                "severity": "critical",
                "description": (
                    "Spark driver pod fl32cfad0d758f29068a909050374e0d-65f0849eee1af10c-driver "
                    "завершает работу, но Kubernetes logs для pod не находятся"
                ),
            },
        ],
        # Узкое окно: когда наблюдались проблемы и сработали алерты (МСК).
        # Алерты должны попасть ВНУТРЬ этого окна.
        "incident_start": datetime(2026, 6, 22, 9, 50, 0, tzinfo=MSK),
        "incident_end":   datetime(2026, 6, 22, 10, 5, 0, tzinfo=MSK),

        # Широкое окно: откуда грузить логи (контекст вокруг инцидента, МСК).
        # По умолчанию ±1 час от incident window. Можно убрать — тогда == incident.
        "context_start": datetime(2026, 6, 22, 9, 45, 0, tzinfo=MSK),
        "context_end":   datetime(2026, 6, 22, 10, 10, 0, tzinfo=MSK),
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
#
#  "context"   — то же окно FREEFORM_START..FREEFORM_END, но FREEFORM_ALERTS может
#               быть пустым. Используй, когда есть только комментарий SRE и нужно
#               сделать summary логов + гипотезы по этому контексту.
# ══════════════════════════════════════════════════════════════════════

MODE = "context"   # "incidents" | "freeform" | "context"

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
#  Если FREEFORM_ALERTS заданы, они должны попасть внутрь этого диапазона.
#  Для MODE="context" FREEFORM_ALERTS можно оставить пустым.
#  FREEFORM_CONTEXT — описание для LLM: что случилось, что ищем.
# ══════════════════════════════════════════════════════════════════════

FREEFORM_START:   datetime | None = datetime(2026, 6, 22, 9, 45, 0, tzinfo=MSK)
FREEFORM_END:     datetime | None = datetime(2026, 6, 22, 10, 10, 0, tzinfo=MSK)
FREEFORM_CONTEXT: str             = """
    SRE-комментарий:
    fl32cfad0d758f29068a909050374e0d-65f0849eee1af10c-driver pod завершает работу,
    при этом в Kubernetes логов pod не обнаруживается.
    Такие кейсы не единичны, происходят периодически, задания могут падать абсолютно разные.
"""
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

# Сколько групп (агрегированных строк) тащить из ClickHouse за один запрос.
# Не влияет на размер промпта — после выборки строки нарезаются по токенам.
BATCH_SIZE = 1000

# Множитель для лимита сырых строк внутри запроса.
# raw_limit = BATCH_SIZE * BATCH_RAW_MULTIPLIER.
# Чем выше повторяемость логов — тем больше нужен множитель чтобы гарантировать
# BATCH_SIZE групп на странице. 50 покрывает компрессию до 50:1.
BATCH_RAW_MULTIPLIER = 50

# Максимум токенов на один MAP-батч (размер одного чанка логов в промпте).
# None → автоматически 55% от MAX_CONTEXT_TOKENS.
# Задай явно если нужен контроль: например 40_000 при слабой модели,
# или 80_000 если хочешь меньше батчей при большом контексте.
MAX_BATCH_TOKENS = None

# Максимум событий (events) в MergedAnalysis после каждого REDUCE-шага.
# При превышении самые незначительные (по severity) отбрасываются.
MAX_EVENTS_PER_MERGE = 30

# Разбивка периода загрузки на временны́е слайсы (часов на один слайс).
# Актуально только если SQL использует BETWEEN по периоду в innermost-запросе.
# При LIMIT на сырых строках (текущий шаблон) слайсы не нужны — ставь 0.
QUERY_TIME_SLICE_HOURS = 0

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
        logs_sql_templates=[s.strip() for s in LOGS_SQLS],
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
        batch_raw_multiplier=BATCH_RAW_MULTIPLIER,
        max_events_per_merge=MAX_EVENTS_PER_MERGE,
        max_batch_tokens=MAX_BATCH_TOKENS,
        query_time_slice_hours=QUERY_TIME_SLICE_HOURS,
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

    Если incident_end совпадал с context_end (как в freeform-режиме) —
    обрезаем оба, чтобы не нарушить validate_windows().
    В incidents-режиме они обычно различаются, поэтому incident_end не трогаем.
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
    # В freeform incident_end == context_end — обрезаем оба чтобы не нарушить validate_windows
    if incident.get("incident_end") == ctx_end:
        result["incident_end"] = trim_end
    return result


async def main(only: str | None) -> None:
    from log_summarizer.utils.logging import setup_pipeline_logging

    # ── Валидация SQL-шаблонов ────────────────────────────────────────
    _validate_sql_templates(LOGS_SQLS)

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

    # ── Freeform/context mode ─────────────────────────────────────────
    if MODE in {"freeform", "context"}:
        if not FREEFORM_START or not FREEFORM_END:
            raise SystemExit(
                f"FREEFORM_START и FREEFORM_END должны быть заданы для MODE=\"{MODE}\".\n"
                "Заполни их в run_pipeline.py."
            )
        # Проверяем что все заданные алерты попадают в период. В MODE="context"
        # список может быть пустым: тогда пайплайн делает summary логов по SRE-комментарию.
        for a in FREEFORM_ALERTS:
            ft = a.get("fired_at")
            if ft and not (FREEFORM_START <= ft <= FREEFORM_END):
                raise SystemExit(
                    f"Алерт '{a['name']}' fired_at={ft} выходит за пределы "
                    f"FREEFORM_START..FREEFORM_END ({FREEFORM_START} — {FREEFORM_END})."
                )
        name = f"{MODE}-" + FREEFORM_START.strftime("%Y-%m-%dT%H-%M")
        incident = {
            "name": name,
            "context": FREEFORM_CONTEXT or (
                f"Анализ логов за период {FREEFORM_START} — {FREEFORM_END}. "
                "Явных алертов нет; используй только текстовый контекст и сами логи."
            ),
            "alerts": FREEFORM_ALERTS,
            "incident_start": FREEFORM_START,
            "incident_end":   FREEFORM_END,
            "context_start":  FREEFORM_START,
            "context_end":    FREEFORM_END,
        }
        if AUTO_TRIM_AFTER_LAST_ALERT:
            incident = _apply_trim(incident, log)
        log.info("%s mode: %s → %s  ·  alerts=%d", MODE.capitalize(), FREEFORM_START, FREEFORM_END, len(FREEFORM_ALERTS))
        name, result, orchestrator = await run_incident(ch, incident, run_dir)
        if isinstance(result, Exception):
            raise SystemExit(f"Пайплайн завершился с ошибкой: {result}")
        artifact_dir = orchestrator.run_dir if orchestrator is not None else None
        if artifact_dir is not None:
            mono_path = artifact_dir / "report.md"
            mono_path.write_text(result, encoding="utf-8")
            log.info("report.md → %s", mono_path.resolve())
        log.info("%s mode завершён. Артефакты: %s", MODE.capitalize(), run_dir.resolve())
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


async def resume(map_dir_path: str, incident_name: str | None) -> None:
    """Продолжает пайплайн с REDUCE-фазы, загружая сохранённые MAP-чанки.

    Args:
        map_dir_path: Путь к папке map/ (или к artifact_dir — map/ найдём сами).
        incident_name: Имя инцидента из INCIDENTS для восстановления конфига.
                       Если None — используется первый инцидент из списка.
    """
    from log_summarizer.config import PipelineConfig
    from log_summarizer.models import make_alerts
    from log_summarizer.orchestrator import PipelineOrchestrator
    from log_summarizer.utils.logging import setup_pipeline_logging
    from pathlib import Path as _Path

    # Ищем папку map/ — может быть передана напрямую или как artifact_dir
    map_dir = _Path(map_dir_path)
    if not (map_dir / "chunk_000.json").exists() and (map_dir / "map").is_dir():
        map_dir = map_dir / "map"
    if not map_dir.is_dir():
        raise SystemExit(f"Папка не найдена: {map_dir}")

    artifact_dir = map_dir.parent
    log_path = artifact_dir / "pipeline_resume.log"
    setup_pipeline_logging(LOG_LEVEL, log_file=str(log_path))
    log = logging.getLogger("run_pipeline.resume")
    log.info("Лог resume → %s", log_path.resolve())

    # Находим конфиг инцидента
    if MODE == "freeform":
        incident = {
            "name": "freeform-resume",
            "context": FREEFORM_CONTEXT or "",
            "alerts": FREEFORM_ALERTS,
            "incident_start": FREEFORM_START,
            "incident_end":   FREEFORM_END,
            "context_start":  FREEFORM_START,
            "context_end":    FREEFORM_END,
        }
    else:
        candidates = [i for i in INCIDENTS if incident_name is None or i["name"] == incident_name]
        if not candidates:
            raise SystemExit(f"Инцидент '{incident_name}' не найден в INCIDENTS")
        incident = candidates[0]

    log.info("RESUME: инцидент=%s  map_dir=%s", incident["name"], map_dir.resolve())

    config = PipelineConfig(
        logs_sql_templates=[s.strip() for s in LOGS_SQLS],
        incident_context=incident["context"].strip(),
        incident_start=incident.get("incident_start") or incident.get("start"),
        incident_end=incident.get("incident_end") or incident.get("end"),
        context_start=incident.get("context_start"),
        context_end=incident.get("context_end"),
        alerts=make_alerts(incident.get("alerts", [])),
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        model_supports_tool_calling=MODEL_SUPPORTS_TOOL_CALLING,
        runs_dir="",  # не создаём новую папку — пишем в artifact_dir
    )

    # Подключаемся к ClickHouse (нужен для инициализации оркестратора)
    try:
        import clickhouse_connect
        ch = clickhouse_connect.get_client(
            host=CH_HOST, port=CH_PORT,
            username=CH_USER, password=CH_PASSWORD,
            database=CH_DATABASE,
        )
    except Exception as exc:
        raise SystemExit(f"ClickHouse: {exc}")

    orchestrator = PipelineOrchestrator(ch, config)
    # Перенаправляем артефакты в ту же папку что была при первом запуске
    orchestrator._run_dir = artifact_dir
    orchestrator.run_dir  = artifact_dir
    orchestrator._multipass_generator._run_dir = artifact_dir
    orchestrator._report_generator._run_dir    = artifact_dir

    report = await orchestrator.run_from_map_dir(map_dir)

    report_path = artifact_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("report.md → %s", report_path.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default=None, metavar="NAME",
                   help="Запустить только один инцидент по имени (только режим incidents)")
    p.add_argument("--list", action="store_true",
                   help="Показать список инцидентов и выйти")
    p.add_argument(
        "--resume", default=None, metavar="MAP_DIR",
        help=(
            "Продолжить с REDUCE, загрузив MAP-чанки из указанной папки. "
            "Пример: --resume runs/2026-04-01T10-00-00/flex-spark-.../2026-04-01T10-05-00"
        ),
    )
    p.add_argument("--resume-incident", default=None, metavar="NAME",
                   help="Имя инцидента для --resume (если несколько в INCIDENTS)")
    args = p.parse_args()

    if args.list:
        if MODE == "freeform":
            print(f"  freeform  {FREEFORM_START} → {FREEFORM_END}")
        else:
            for i in INCIDENTS:
                s = i.get("incident_start") or i.get("start", "?")
                e = i.get("incident_end")   or i.get("end",   "?")
                print(f"  {i['name']}  {s} → {e}")
    elif args.resume:
        asyncio.run(resume(args.resume, args.resume_incident))
    else:
        asyncio.run(main(args.only))
