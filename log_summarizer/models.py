"""Pydantic-модели для MAP / REDUCE / Evidence / Report фаз."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════
#  Severity enum
# ══════════════════════════════════════════════════════════════════════

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @classmethod
    def priority(cls, value: "Severity") -> int:
        """Числовой приоритет для сортировки (меньше = важнее)."""
        _order = [cls.CRITICAL, cls.HIGH, cls.MEDIUM, cls.LOW, cls.INFO]
        try:
            return _order.index(value)
        except ValueError:
            return len(_order)


# ══════════════════════════════════════════════════════════════════════
#  Алерты — структурированный вход, ID назначается автоматически
# ══════════════════════════════════════════════════════════════════════

class Alert(BaseModel):
    """Алерт, сработавший во время инцидента.

    Создаётся через make_alerts() — ID присваивается автоматически.
    Передаётся в промпт MAP-фазы, чтобы LLM мог проставить статус по каждому.
    """

    id: str                             # "alert-001", "alert-002", ... — авто
    name: str                           # e.g. "AirflowKubernetesExecutorFailed"
    fired_at: Optional[datetime] = None
    severity: Severity = Severity.HIGH
    description: Optional[str] = None  # текст алерта или краткое пояснение


class AlertStatus(str, Enum):
    """Статус объяснённости алерта в контексте батча / итогового анализа."""

    EXPLAINED = "explained"         # причина найдена в логах этого батча
    PARTIAL = "partial"             # частично объяснён, но не полностью
    NOT_EXPLAINED = "not_explained" # алерт виден, но логи не объясняют
    NOT_SEEN = "not_seen"           # в этом батче нет данных об алерте

    @classmethod
    def priority(cls, value: "AlertStatus") -> int:
        """Приоритет для программного merge (меньше = лучше объяснён)."""
        _order = [cls.EXPLAINED, cls.PARTIAL, cls.NOT_EXPLAINED, cls.NOT_SEEN]
        try:
            return _order.index(value)
        except ValueError:
            return len(_order)


class AlertRef(BaseModel):
    """Ссылка на алерт с оценкой его объяснённости в данном батче/анализе."""

    alert_id: str                       # ссылка на Alert.id
    status: AlertStatus
    comment: Optional[str] = None       # краткое пояснение (≤80 символов)


def make_alerts(raw: list[dict]) -> list[Alert]:
    """Создаёт список Alert из сырых dict, авто-присваивая ID.

    Пример входа:
        [{"name": "AirflowKubernetesExecutorFailed",
          "fired_at": datetime(...), "severity": "critical"}]

    Присваивает: alert-001, alert-002, ...
    """
    result: list[Alert] = []
    for i, item in enumerate(raw, 1):
        result.append(Alert(
            id=f"alert-{i:03d}",
            name=item["name"],
            fired_at=item.get("fired_at"),
            severity=Severity(item.get("severity", "high")),
            description=item.get("description"),
        ))
    return result


# ══════════════════════════════════════════════════════════════════════
#  Входные структуры (загрузка из ClickHouse)
# ══════════════════════════════════════════════════════════════════════

class LogRow(BaseModel):
    """Одна строка лога из ClickHouse."""

    timestamp: datetime
    level: Optional[str] = None
    source: Optional[str] = None   # сервис / контейнер / pod
    message: str
    raw_line: str                  # оригинальная строка целиком


class MetricRow(BaseModel):
    """Одна точка метрики из ClickHouse."""

    timestamp: datetime
    service: str
    metric_name: str               # e.g. "p99_latency_ms", "error_rate"
    value: float


class Chunk(BaseModel):
    """Нарезанный кусок строк лога — входная единица MAP."""

    rows: list[LogRow]
    time_range: tuple[datetime, datetime]
    token_estimate: int


# ══════════════════════════════════════════════════════════════════════
#  Evidence — несжимаемые дословные цитаты
# ══════════════════════════════════════════════════════════════════════

class Evidence(BaseModel):
    """Дословная цитата из лога.

    Никогда не проходит через LLM-сжатие — только конкатенация и дедуп.
    Хранится в evidence_bank отдельно от нарратива.
    """

    id: str                              # e.g. "ev-001"
    timestamp: datetime
    source: str
    raw_line: str                        # точная строка из лога
    severity: Severity
    linked_event_id: Optional[str] = None  # ссылка на Event.id


# ══════════════════════════════════════════════════════════════════════
#  MAP-фаза — вывод LLM
# ══════════════════════════════════════════════════════════════════════

class Event(BaseModel):
    """Ключевое событие, выделенное из батча."""

    id: str                              # e.g. "evt-007-001"
    timestamp: datetime
    source: str
    description: str                     # краткое описание своими словами
    severity: Severity                   # объективная серьёзность события
    importance: float = 0.5             # 0.0-1.0: релевантность для данного расследования
    tags: list[str] = Field(default_factory=list)  # oom/connection/timeout/...


class Hypothesis(BaseModel):
    """Гипотеза о причине инцидента."""

    id: str
    title: str
    description: str
    confidence: str                      # "low" | "medium" | "high"
    supporting_event_ids: list[str]      # ссылки на Event.id
    contradicting_event_ids: list[str] = Field(default_factory=list)
    related_alert_ids: list[str] = Field(default_factory=list)  # ссылки на Alert.id


class Anomaly(BaseModel):
    """Что-то необычное, замеченное в логах."""

    description: str
    related_event_ids: list[str] = Field(default_factory=list)


class BatchAnalysis(BaseModel):
    """Результат MAP-фазы: анализ одного батча логов."""

    time_range: tuple[datetime, datetime]
    narrative: str                       # 3-5 предложений: что происходило
    events: list[Event]
    evidence: list[Evidence]             # дословные строки из логов
    hypotheses: list[Hypothesis]
    anomalies: list[Anomaly]
    metrics_context: Optional[str] = None   # что показывали метрики
    data_quality: Optional[str] = None     # "шумные логи" / "пустой батч" / ...
    alert_refs: list[AlertRef] = Field(default_factory=list)            # статус по каждому алерту
    preliminary_recommendations: list[str] = Field(default_factory=list)  # ранние рекомендации

    def to_json_str(self) -> str:
        """Сериализация для REDUCE-промптов.

        Исключает evidence (→ evidence_bank) и alert_refs (→ программный merge).
        """
        return self.model_copy(
            update={"evidence": [], "alert_refs": []}
        ).model_dump_json()


# ══════════════════════════════════════════════════════════════════════
#  REDUCE-фаза — вывод LLM
# ══════════════════════════════════════════════════════════════════════

class CausalLink(BaseModel):
    """Причинно-следственная связь между событиями из разных батчей."""

    from_event_id: str
    to_event_id: str
    description: str                     # "OOM в payments вызван ростом запросов из api-gateway"
    confidence: str                      # "low" | "medium" | "high"


class TimeGap(BaseModel):
    """Разрыв в данных или необъяснённый промежуток."""

    start: datetime
    end: datetime
    description: str


class MergedAnalysis(BaseModel):
    """Результат REDUCE-фазы: объединённый анализ."""

    time_range: tuple[datetime, datetime]
    narrative: str                       # связный нарратив всего инцидента
    events: list[Event]                  # дедуплицированный timeline
    causal_chains: list[CausalLink]      # причинно-следственные цепочки
    hypotheses: list[Hypothesis]         # уточнённые гипотезы
    anomalies: list[Anomaly]
    gaps: list[TimeGap]                  # дыры в данных
    impact_summary: str                  # какие сервисы, как долго, масштаб
    preliminary_recommendations: list[str] = Field(default_factory=list)  # из MAP-фазы

    # Прикрепляются в конце REDUCE — НЕ проходят через LLM
    evidence_bank: list[Evidence] = Field(default_factory=list)
    alert_refs: list[AlertRef] = Field(default_factory=list)

    def to_json_str(self) -> str:
        """Сериализация для REDUCE-промптов.

        Исключает evidence_bank и alert_refs — они обрабатываются отдельно.
        """
        return self.model_copy(
            update={"evidence_bank": [], "alert_refs": []}
        ).model_dump_json()

    def to_json_str_with_evidence(self) -> str:
        """Полная сериализация включая evidence_bank."""
        return self.model_dump_json()


# ══════════════════════════════════════════════════════════════════════
#  ReportBudget — бюджетирование финального отчёта
# ══════════════════════════════════════════════════════════════════════

class ReportBudget(BaseModel):
    """Бюджет токенов для каждого компонента финального отчёта."""

    analysis_tokens: int    # бюджет на MergedAnalysis
    evidence_tokens: int    # бюджет на evidence_bank
    early_tokens: int       # бюджет на early_summaries
