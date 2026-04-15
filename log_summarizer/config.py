"""PipelineConfig — конфигурация всего пайплайна Log Summarizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from log_summarizer.models import Alert


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна анализа логов.

    Все настройки собраны в одном месте. Создаётся один раз
    и передаётся во все модули через конструктор.
    """

    # ── SQL-шаблоны ──────────────────────────────────────────────────
    # Плейсхолдеры: {start_time}, {end_time}, {limit}
    logs_sql_template: str = ""

    # Опционально: SQL для метрик сервисов
    metrics_sql_template: Optional[str] = None

    # ── Контекст инцидента ────────────────────────────────────────────
    # Описание инцидента словами: что случилось, какие сервисы затронуты.
    incident_context: str = ""
    incident_start: Optional[datetime] = None
    incident_end: Optional[datetime] = None

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

    # ── Параметры REDUCE ─────────────────────────────────────────────
    max_group_size: int = 4         # макс элементов в REDUCE-группе
    max_item_chars: int = 40_000    # лимит на 1 item перед merge
    compression_target_pct: int = 50    # % сжатия при overflow
    max_reduce_rounds: int = 15     # максимум раундов дерева
    max_events_per_merge: int = 30  # макс events после каждого merge

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

    def map_batch_tokens(self) -> int:
        """Бюджет токенов на один MAP-вызов (55% от контекста)."""
        return int(self.max_context_tokens * 0.55)

    def reduce_budget_tokens(self) -> int:
        """Бюджет токенов на входные саммари в одном REDUCE-вызове (55%)."""
        return int(self.max_context_tokens * 0.55)

    def incident_start_iso(self) -> str:
        """ISO-строка начала инцидента для промптов."""
        if self.incident_start is None:
            return ""
        return self.incident_start.isoformat()

    def incident_end_iso(self) -> str:
        """ISO-строка конца инцидента для промптов."""
        if self.incident_end is None:
            return ""
        return self.incident_end.isoformat()
