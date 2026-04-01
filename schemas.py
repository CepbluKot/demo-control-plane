"""
Pydantic-схемы для map/reduce summary.
Одна и та же структура используется на всех уровнях pipeline:
map-выход, reduce-выход на каждом уровне, финальное summary.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ── Раздел 1: Context ──────────────────────────────────────────────

class Context(BaseModel):
    """Метаданные батча. Большинство полей заполняется программно до LLM-вызова."""

    batch_id: str = Field(
        description="Уникальный идентификатор батча в pipeline"
    )
    time_range_start: str = Field(
        description="Начало временного окна батча, ISO 8601 UTC"
    )
    time_range_end: str = Field(
        description="Конец временного окна батча, ISO 8601 UTC"
    )
    total_log_entries: int = Field(
        ge=0,
        description="Количество строк логов в батче после препроцессинга"
    )
    source_query: list[str] = Field(
        default_factory=list,
        description="SQL-запросы, которыми забирались данные для этого батча"
    )
    source_services: list[str] = Field(
        default_factory=list,
        description="Уникальные сервисы/хосты, чьи логи есть в батче"
    )


# ── Раздел 2: Timeline ─────────────────────────────────────────────

class TimelineEvent(BaseModel):
    """Одно значимое событие, извлечённое из логов."""

    id: str = Field(description="Уникальный id события, например 'evt-001'")
    timestamp: str = Field(
        description="Дословно из лога, ISO 8601 с максимальной точностью"
    )
    source: str = Field(description="Сервис или хост, породивший запись")
    description: str = Field(
        description="Краткое описание своими словами, 1-2 предложения"
    )
    severity: Literal["critical", "high", "medium", "low"]
    importance: float = Field(
        ge=0.0, le=1.0,
        description="Релевантность для расследования. >0.7 = pinned, не отбрасывается при сжатии"
    )
    evidence_type: Literal["FACT", "HYPOTHESIS"]
    evidence_quote: Optional[str] = Field(
        default=None,
        description="Дословная цитата из лога. Только для FACT"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Метки: deployment, OOM, network, latency, timeout и т.д."
    )

    @model_validator(mode="after")
    def fact_must_have_quote(self) -> "TimelineEvent":
        if self.evidence_type == "FACT" and not self.evidence_quote:
            raise ValueError("FACT events must have evidence_quote")
        if self.evidence_type == "HYPOTHESIS" and self.evidence_quote:
            raise ValueError("HYPOTHESIS events must not have evidence_quote")
        return self


# ── Раздел 3: Causal Links ─────────────────────────────────────────

class CausalLink(BaseModel):
    """Причинно-следственная связь: событие A → событие B."""

    id: str
    cause_event_id: str = Field(description="id события-причины из timeline")
    effect_event_id: str = Field(description="id события-следствия из timeline")
    mechanism: str = Field(
        description="КАК именно причина привела к следствию. Конкретный механизм, не просто 'A вызвало B'"
    )
    confidence: float = Field(ge=0.0, le=1.0)


# ── Раздел 4: Alert Refs ───────────────────────────────────────────

class AlertRef(BaseModel):
    """Привязка алерта из UI к данным батча."""

    alert_id: str = Field(description="Идентификатор алерта из пользовательского контекста, дословно")
    status: Literal["EXPLAINED", "PARTIALLY", "NOT_EXPLAINED", "NOT_SEEN_IN_BATCH"]
    related_events: list[str] = Field(
        default_factory=list,
        description="Список id из timeline, связанных с алертом"
    )
    explanation: str = Field(
        default="",
        description="Почему такой статус, 1-2 предложения"
    )


# ── Раздел 5: Hypotheses ───────────────────────────────────────────

class Hypothesis(BaseModel):
    """Гипотеза первопричины."""

    id: str
    related_alert_ids: list[str] = Field(
        description="К каким алертам из UI относится гипотеза"
    )
    title: str = Field(description="Короткая формулировка, одна строка")
    description: str = Field(
        description="Развёрнутое обоснование, 2-5 предложений"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_events: list[str] = Field(
        default_factory=list,
        description="id из timeline, подтверждающие гипотезу"
    )
    contradicting_events: list[str] = Field(
        default_factory=list,
        description="id из timeline, ставящие гипотезу под сомнение"
    )
    status: Literal["active", "merged", "conflicting", "dismissed"] = "active"


# ── Раздел 6: Pinned Facts ─────────────────────────────────────────

class PinnedFact(BaseModel):
    """Критичный контекстный факт, не событие, а статичная информация."""

    id: str
    fact: str = Field(description="Дословная формулировка факта")
    evidence_quote: str = Field(description="Цитата из лога, откуда извлечён")
    relevance: str = Field(
        description="К чему относится: алерт, гипотеза, аспект расследования"
    )
    importance: float = Field(ge=0.0, le=1.0)


# ── Раздел 7: Gaps ─────────────────────────────────────────────────

class Gap(BaseModel):
    """Разрыв в данных или причинно-следственной цепочке."""

    id: str
    description: str = Field(
        description="Что именно неизвестно, конкретно"
    )
    between_events: list[str] = Field(
        default_factory=list,
        description="Пара id из timeline, между которыми разрыв. Может быть пустым"
    )
    missing_data: str = Field(
        description="Какие данные нужны для закрытия разрыва, конкретно"
    )


# ── Раздел 8: Impact ───────────────────────────────────────────────

class DegradationPeriod(BaseModel):
    start: str = Field(description="Начало деградации, ISO 8601")
    end: str = Field(description="Конец деградации, ISO 8601")


class Impact(BaseModel):
    """Оценка масштаба и влияния инцидента."""

    affected_services: list[str] = Field(default_factory=list)
    affected_operations: list[str] = Field(
        default_factory=list,
        description="Пользовательские сценарии: авторизация, платежи и т.д."
    )
    error_counts: list[str] = Field(
        default_factory=list,
        description="Количественные данные: 'HTTP 503 — 1247 за 3 мин' и т.д."
    )
    degradation_period: Optional[DegradationPeriod] = None


# ── Раздел 9: Conflicts ────────────────────────────────────────────

class ConflictSide(BaseModel):
    description: str
    supporting_events: list[str] = Field(default_factory=list)


class Conflict(BaseModel):
    """Противоречивые сигналы внутри батча."""

    id: str
    description: str = Field(description="Суть противоречия")
    side_a: ConflictSide
    side_b: ConflictSide
    resolution: Optional[str] = Field(
        default=None,
        description="Разрешение конфликта, если возможно. Иначе None"
    )


# ── Раздел 10: Data Quality ────────────────────────────────────────

class DataQuality(BaseModel):
    """Оценка качества и полноты данных."""

    is_empty: bool = Field(
        default=False,
        description="True если timeline пуст — нет значимых событий"
    )
    noise_ratio: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Доля бесполезных записей. 0 = всё полезно, 1 = всё шум"
    )
    notes: str = Field(
        default="",
        description="Свободный текст: проблемы с данными, подозрения"
    )


# ── Раздел 11: Preliminary Recommendations ─────────────────────────

class Recommendation(BaseModel):
    """Предварительная рекомендация для SRE."""

    id: str
    priority: Literal["P0", "P1", "P2"]
    action: str = Field(
        description="Конкретное действие, actionable без дополнительных уточнений"
    )
    rationale: str = Field(
        description="Обоснование со ссылками на события из timeline"
    )
    related_hypothesis_ids: list[str] = Field(default_factory=list)


# ── Корневая модель ────────────────────────────────────────────────

class IncidentSummary(BaseModel):
    """
    Корневая модель map/reduce summary.
    Используется как response_model в Instructor на всех уровнях pipeline.
    """

    context: Context
    timeline: list[TimelineEvent] = Field(default_factory=list)
    causal_links: list[CausalLink] = Field(default_factory=list)
    alert_refs: list[AlertRef] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    pinned_facts: list[PinnedFact] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    impact: Impact = Field(default_factory=Impact)
    conflicts: list[Conflict] = Field(default_factory=list)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    preliminary_recommendations: list[Recommendation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_refs(self) -> "IncidentSummary":
        """Проверяет, что все ссылки на event_id указывают на существующие события."""
        valid_ids = {e.id for e in self.timeline}

        for link in self.causal_links:
            if link.cause_event_id not in valid_ids:
                raise ValueError(
                    f"causal_link {link.id}: cause_event_id '{link.cause_event_id}' "
                    f"not found in timeline"
                )
            if link.effect_event_id not in valid_ids:
                raise ValueError(
                    f"causal_link {link.id}: effect_event_id '{link.effect_event_id}' "
                    f"not found in timeline"
                )

        for ref in self.alert_refs:
            for eid in ref.related_events:
                if eid not in valid_ids:
                    raise ValueError(
                        f"alert_ref {ref.alert_id}: related_event '{eid}' "
                        f"not found in timeline"
                    )

        for hyp in self.hypotheses:
            for eid in hyp.supporting_events:
                if eid not in valid_ids:
                    raise ValueError(
                        f"hypothesis {hyp.id}: supporting_event '{eid}' "
                        f"not found in timeline"
                    )
            for eid in hyp.contradicting_events:
                if eid not in valid_ids:
                    raise ValueError(
                        f"hypothesis {hyp.id}: contradicting_event '{eid}' "
                        f"not found in timeline"
                    )

        for gap in self.gaps:
            for eid in gap.between_events:
                if eid not in valid_ids:
                    raise ValueError(
                        f"gap {gap.id}: between_event '{eid}' "
                        f"not found in timeline"
                    )

        for conflict in self.conflicts:
            for eid in conflict.side_a.supporting_events:
                if eid not in valid_ids:
                    raise ValueError(
                        f"conflict {conflict.id} side_a: event '{eid}' "
                        f"not found in timeline"
                    )
            for eid in conflict.side_b.supporting_events:
                if eid not in valid_ids:
                    raise ValueError(
                        f"conflict {conflict.id} side_b: event '{eid}' "
                        f"not found in timeline"
                    )

        return self
