"""PipelineConfig — конфигурация всего пайплайна Log Summarizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

# Московское время — UTC+3, переход на летнее время в России отменён.
MSK = timezone(timedelta(hours=3))

from log_summarizer.models import Alert


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна анализа логов.

    Все настройки собраны в одном месте. Создаётся один раз
    и передаётся во все модули через конструктор.
    """

    # ── SQL-шаблоны ──────────────────────────────────────────────────
    # Список SQL-шаблонов для загрузки логов. Первый элемент — основной источник,
    # остальные — дополнительные (например, log_k8s_events). DataLoader запрашивает
    # все источники независимо с одним last_ts, результаты сортируются в Python.
    # Обязательные плейсхолдеры: {last_ts}, {period_end}, {limit}, {raw_limit}.
    logs_sql_templates: list = field(default_factory=list)

    # Опционально: SQL для метрик сервисов
    metrics_sql_template: Optional[str] = None

    # ── Контекст инцидента ────────────────────────────────────────────
    # Описание инцидента словами: что случилось, какие сервисы затронуты.
    incident_context: str = ""

    # Узкое окно: когда сработали алерты / наблюдались проблемы.
    # Используется для zone-разметки логов и фокуса анализа.
    incident_start: Optional[datetime] = None
    incident_end: Optional[datetime] = None

    # Широкое окно: откуда грузить логи (контекст вокруг инцидента).
    # Если не задано — автоматически расширяется на context_auto_expand_hours в каждую сторону.
    # Пример: инцидент в 14:15-14:35, контекст 13:15-15:35 (±1 час).
    # Задать явно чтобы переопределить авто-расширение.
    context_start: Optional[datetime] = None
    context_end: Optional[datetime] = None

    # Часов расширения в каждую сторону от incident_start/end если context не задан явно.
    # 0 — отключить авто-расширение (context == incident).
    context_auto_expand_hours: float = 1.0

    # Алерты, сработавшие во время инцидента. Создаются через make_alerts().
    # MAP-фаза проставляет статус по каждому алерту; REDUCE мержит программно.
    alerts: list[Alert] = field(default_factory=list)

    # ── LLM ──────────────────────────────────────────────────────────
    model: str = "default-model"
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "sk-placeholder"
    max_context_tokens: int = 150_000

    # ── Параметры MAP ────────────────────────────────────────────────
    batch_size: int = 200           # верхняя граница строк на батч
    map_concurrency: int = 5        # параллельных MAP-вызовов
    max_split_depth: int = 6        # максимум рекурсивных делений
    min_batch_lines: int = 20       # минимум строк при split

    # Максимум токенов на один MAP-батч.
    # None → вычисляется автоматически как 55% от max_context_tokens.
    # Задай явно если хочешь контролировать размер батча независимо от контекста.
    max_batch_tokens: Optional[int] = None

    # ── Параметры REDUCE ─────────────────────────────────────────────
    max_group_size: int = 4         # макс элементов в REDUCE-группе
    max_item_chars: int = 40_000    # лимит на 1 item перед merge
    compression_target_pct: int = 50    # % сжатия при overflow
    max_reduce_rounds: int = 15     # максимум раундов дерева
    max_events_per_merge: int = 30  # макс events после каждого merge

    # Порог превентивной компрессии перед merge (суммарный размер входного payload в символах).
    # Если sum(len(item_json) for item in group) > этого значения — сначала сжимаем входы,
    # потом отправляем в LLM. 0 = отключить превентивную компрессию.
    # Подбирается эмпирически: если при ~60000 символов стабильно 400 — ставь 50000.
    pre_compress_threshold: int = 50_000

    # ── Параметры финального отчёта ──────────────────────────────────
    early_summaries_budget_chars: int = 40_000

    # Доли контекста под каждый компонент
    report_budget_analysis_pct: float = 0.50
    report_budget_evidence_pct: float = 0.30
    report_budget_early_pct: float = 0.20

    # Резерв токенов на ответ LLM (отчёт объёмный)
    report_response_reserve_tokens: int = 30_000
    # Оценка размера system prompt
    report_system_prompt_tokens: int = 3_000

    # ── Retry / timeouts ─────────────────────────────────────────────
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # 2, 4, 8 секунд

    # ── Temperature ──────────────────────────────────────────────────
    temperature_map: float = 0.2
    temperature_reduce: float = 0.2
    temperature_report: float = 0.3

    # ── Instructor (структурированный вывод) ─────────────────────────
    use_instructor: bool = True
    model_supports_tool_calling: bool = False  # False = JSON mode

    # ── Наблюдаемость ────────────────────────────────────────────────
    # Каждый прогон создаёт подпапку runs/{timestamp}/ с промптами,
    # промежуточными результатами и финальным отчётом.
    # Пустая строка = не сохранять артефакты.
    runs_dir: str = "runs"

    # Заполняется оркестратором после загрузки данных — число строк логов.
    # Отображается в секции 1 отчёта.
    total_log_rows: int = 0

    # Разбивка периода загрузки на временны́е слайсы.
    # Нужно если SQL содержит оконные функции (lagInFrame, sum OVER) и период длинный —
    # ClickHouse держит весь диапазон {start_time}..{end_time} в памяти для их вычисления.
    # None / 0 — без разбивки (один запрос на весь период).
    # Рекомендовано: 2.0 для периодов > 4 часов с оконными функциями в SQL.
    query_time_slice_hours: float = 0.0

    # Множитель для лимита сырых строк во внутреннем подзапросе.
    # raw_limit = batch_size * batch_raw_multiplier.
    # Нужен чтобы GROUP BY всегда имел достаточно сырых строк для формирования
    # batch_size групп даже при высокой степени повторяемости логов.
    # Пример: batch_size=1000, multiplier=50 → обрабатываем до 50k сырых строк.
    batch_raw_multiplier: int = 50

    # ── Вспомогательные методы ───────────────────────────────────────

    def context_start_actual(self) -> Optional[datetime]:
        """Фактическое начало окна загрузки логов.

        Приоритет: явный context_start → авто-расширение на N часов → incident_start.
        """
        if self.context_start is not None:
            return self.context_start
        if self.incident_start is not None and self.context_auto_expand_hours > 0:
            return self.incident_start - timedelta(hours=self.context_auto_expand_hours)
        return self.incident_start

    def context_end_actual(self) -> Optional[datetime]:
        """Фактическое окончание окна загрузки логов.

        Приоритет: явный context_end → авто-расширение на N часов → incident_end.
        """
        if self.context_end is not None:
            return self.context_end
        if self.incident_end is not None and self.context_auto_expand_hours > 0:
            return self.incident_end + timedelta(hours=self.context_auto_expand_hours)
        return self.incident_end

    def has_context_window(self) -> bool:
        """True если окно загрузки шире окна инцидента."""
        cs = self.context_start_actual()
        ce = self.context_end_actual()
        return (cs != self.incident_start) or (ce != self.incident_end)

    def validate_windows(self) -> list[str]:
        """Проверяет согласованность временных окон. Возвращает список ошибок."""
        errors: list[str] = []
        cs = self.context_start_actual()
        ce = self.context_end_actual()
        ins = self.incident_start
        ine = self.incident_end

        if cs and ce and cs > ce:
            errors.append(f"context_start ({cs}) > context_end ({ce})")
        if ins and ine and ins > ine:
            errors.append(f"incident_start ({ins}) > incident_end ({ine})")
        if cs and ins and cs > ins:
            errors.append(f"context_start ({cs}) > incident_start ({ins})")
        if ce and ine and ine > ce:
            errors.append(f"incident_end ({ine}) > context_end ({ce})")

        # Проверяем алерты внутри incident window
        for alert in self.alerts:
            if alert.fired_at and ins and ine:
                if not (ins <= alert.fired_at <= ine):
                    errors.append(
                        f"Alert '{alert.name}' fired_at={alert.fired_at} "
                        f"outside incident window [{ins}, {ine}]"
                    )
        return errors

    def map_batch_tokens(self) -> int:
        """Бюджет токенов на один MAP-батч.

        Если max_batch_tokens задан явно — используется как есть.
        Иначе: 55% от max_context_tokens (оставшееся — промпт + ответ LLM).
        """
        if self.max_batch_tokens is not None:
            return self.max_batch_tokens
        return int(self.max_context_tokens * 0.55)

    def reduce_budget_tokens(self) -> int:
        """Бюджет токенов на входные саммари в одном REDUCE-вызове (55%)."""
        return int(self.max_context_tokens * 0.55)

    def incident_start_iso(self) -> str:
        """ISO-строка начала инцидента для промптов."""
        return self.incident_start.isoformat() if self.incident_start else ""

    def incident_end_iso(self) -> str:
        """ISO-строка конца инцидента для промптов."""
        return self.incident_end.isoformat() if self.incident_end else ""

    def context_start_iso(self) -> str:
        cs = self.context_start_actual()
        return cs.isoformat() if cs else ""

    def context_end_iso(self) -> str:
        ce = self.context_end_actual()
        return ce.isoformat() if ce else ""
