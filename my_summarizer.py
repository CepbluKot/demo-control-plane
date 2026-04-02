from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Type, TypeVar

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

import prompts as incident_prompts
from alert_merge import merge_alert_refs
from schemas import AlertRef as IncidentAlertRef
from schemas import Conflict as IncidentConflict
from schemas import Context as IncidentContext
from schemas import DataQuality as IncidentDataQuality
from schemas import Impact as IncidentImpact
from schemas import IncidentSummary
from schemas import Recommendation as IncidentRecommendation
from schemas import TimelineEvent as IncidentTimelineEvent
from report_schema import (
    AlertsSection,
    CausalChainsSection,
    ChronologySection,
    ConflictsSection,
    CoverageSection,
    FreeformReport,
    GapDescription,
    GapsSection,
    ImpactSection,
    IncidentReport,
    MetricsReportSection,
    MetricsSection,
    HypothesesSection,
    ReportPartAnalytical,
    ReportPartDescriptive,
    RecommendationsSection,
    SummarySection,
    LimitationsSection,
    ReportSummary,
    DataCoverage,
    ChronologyEvent,
    CausalChain,
    AlertExplanation,
    ReportHypothesis,
    ConflictDescription,
    ImpactAssessment,
    ReportRecommendation,
    AnalysisLimitations,
)
from settings import settings

try:
    import instructor  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    instructor = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency (comes transitively with instructor)
    from openai import OpenAI  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


LOGS_SQL_COLUMNS: tuple[str, ...] = ("timestamp", "value")
DEFAULT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "start_time",
    "end_time",
    "cnt",
    "log",
    "message",
    "time",
    "logtag",
    "ext_ClusterEnv",
    "ext_ClusterEventType",
    "ext_ClusterName",
    "kubernetes_pod_name",
    "kubernetes_namespace_name",
    "kubernetes_container_name",
    "kubernetes_docker_id",
    "kubernetes_container_image",
    "container",
    "pod",
    "node",
    "cluster",
    "level",
    "status",
    "value",
)
logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, Dict[str, Any]], None]
_EARLIEST_PERIOD_START = "1970-01-01T00:00:00+00:00"
TModel = TypeVar("TModel", bound=BaseModel)
class DBPageFetcher(Protocol):
    def __call__(
        self,
        *,
        columns: Sequence[str],
        period_start: str,
        period_end: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        ...


class LLMTextCaller(Protocol):
    def __call__(self, prompt: str) -> str:
        ...


@dataclass
class SummarizerConfig:
    page_limit: int = 1000
    llm_chunk_rows: int = 200
    min_llm_chunk_rows: int = 20
    auto_shrink_on_400: bool = True
    max_shrink_rounds: int = 6
    # New algorithm default: fixed-size reduce groups (algorithm (1).md).
    reduce_group_size: int = 3
    max_reduce_rounds: int = 12
    # 0 or negative => no truncation.
    max_cell_chars: int = 0
    # 0 or negative => no truncation.
    max_summary_chars: int = 0
    # 0 or negative => no local reduce prompt-length cap.
    reduce_prompt_max_chars: int = 0
    adaptive_reduce_on_overflow: bool = True
    keep_map_batches_in_memory: bool = False
    keep_map_summaries_in_result: bool = False
    # Number of parallel LLM workers for the MAP phase.
    # 1 = sequential (safe default).  Set to 3-5 for a significant speedup when the
    # LLM endpoint supports concurrent requests — total MAP time becomes roughly
    # ceil(n_batches / map_workers) × avg_llm_latency instead of n_batches × avg_llm_latency.
    map_workers: int = 1
    # New pipeline mode (algorithm (1).md): no local token estimation,
    # split only on real overflow/timeouts + cascading structured reduce.
    use_new_algorithm: bool = False
    # Reduce/compression target size in percents of input size.
    reduce_target_token_pct: int = 50
    compression_target_pct: int = 50
    compression_importance_threshold: float = 0.7
    use_instructor: bool = False
    # If False, instructor works in JSON mode (no tool/function calling).
    # Needed for OpenAI-compatible gateways that do not support tool-calling parser.
    model_supports_tool_calling: bool = True


@dataclass
class SummarizationResult:
    summary: str
    pages_fetched: int
    rows_processed: int
    llm_calls: int
    chunk_summaries: int
    reduce_rounds: int
    map_summaries: List[str]
    map_batches: List[Dict[str, Any]]
    freeform_summary: str = ""


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


DEFAULT_ANTI_HALLUCINATION_RULES = (
    "1) Цитируй источник: timestamp/сообщение/cnt/поле.\n"
    "2) Не додумывай: если данных нет — \"данных недостаточно\".\n"
    "3) Разделяй [ФАКТ] и [ГИПОТЕЗА]; для гипотезы указывай, что проверить.\n"
    "4) Не обобщай сверх данных (не раздувай масштаб).\n"
    "5) Корреляция по времени != причинность.\n"
    "6) Для [ФАКТ]-причинности нужны: A и B в данных, A<=B по времени, механизм влияния, подтверждение механизма.\n"
    "7) В агрегированных логах argMin-поля — пример, cnt — масштаб.\n"
    "8) Не экстраполируй вне временного диапазона данных.\n"
    "9) При противоречиях показывай оба варианта.\n"
    "10) Отделяй [РЕЛЕВАНТНО] от [ФОН]/[НЕЯСНО].\n"
    "11) В хронологии КАЖДОЕ событие обязано содержать полную дату и время (до микросекунд) и timezone."
)


def _chain_section_requirement(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if normalized == "map":
        title = "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЛОКАЛЬНАЯ ЦЕПОЧКА СОБЫТИЙ БАТЧА"
    elif normalized == "reduce":
        title = "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: СВОДНАЯ ЦЕПОЧКА СОБЫТИЙ ИСТОЧНИКА"
    elif normalized == "cross":
        title = "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЕДИНАЯ ЦЕПОЧКА СОБЫТИЙ ИНЦИДЕНТА"
    else:
        title = "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЦЕПОЧКА СОБЫТИЙ"
    return "\n".join(
        [
            title,
            "Оформи красиво и явно как схему в Markdown.",
            "Для КАЖДОГО узла обязательно укажи точный timestamp события: `YYYY-MM-DD HH:MM:SS.ffffff TZ`.",
            "Запрещены абстрактные метки без времени (например, просто t1/t2/t3).",
            "Формат (сохрани стрелки и отступы):",
            "[2026-03-31 12:34:56.123456 MSK] ТРИГГЕР (компонент) [ФАКТ/ГИПОТЕЗА]",
            "    └─> (механизм влияния)",
            "[2026-03-31 12:35:07.654321 MSK] СЛЕДСТВИЕ (компонент) [ФАКТ/ГИПОТЕЗА]",
            "    └─> (механизм влияния)",
            "[2026-03-31 12:35:10.000001 MSK] АЛЕРТ/ПОСЛЕДСТВИЕ [ФАКТ]",
            "Если цепочек несколько — перечисли ЦЕПОЧКА #1, ЦЕПОЧКА #2 и т.д.",
            "Если есть разрыв — вставь узел: [РАЗРЫВ ЦЕПОЧКИ: каких данных не хватает].",
        ]
    )


def _append_chain_requirement(prompt_text: str, stage: str) -> str:
    base = str(prompt_text or "").strip()
    chain_block = _chain_section_requirement(stage)
    incident_link_block = "\n".join(
        [
            "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: СВЯЗЬ С ИНЦИДЕНТОМ ИЗ UI",
            "Нужно явно связать выводы с контекстом, который пользователь ввёл в UI",
            "(incident_description / alerts_list / user goal).",
            "Для каждого пункта инцидента/алерта укажи:",
            "- статус: [ОБЪЯСНЁН] / [ЧАСТИЧНО ОБЪЯСНЁН] / [НЕ ОБЪЯСНЁН]",
            "- доказательства: конкретные timestamp/сообщения/метрики",
            "- причинно-следственная связь (если есть)",
            "- если связи нет: что нужно проверить дополнительно.",
            "Если контекст инцидента пустой — напиши это явно отдельной строкой.",
        ]
    )
    root_cause_hypotheses_block = "\n".join(
        [
            "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ",
            "Сначала выведи подраздел `ИНЦИДЕНТ ИЗ UI (ДОСЛОВНО)`.",
            "В этот подраздел дословно вставь название/описание инцидента из контекста "
            "(incident_description / alerts_list) БЕЗ перефразирования.",
            "Сохрани исходные формулировки, ноды, IP, статусы и время в том виде, как они даны пользователем.",
            "Только после дословного блока переходи к гипотезам первопричин.",
            "Для КАЖДОГО выявленного инцидента/цепочки отдельно перечисли 2-5 гипотез первопричины.",
            "Формат для каждого пункта:",
            "- Инцидент/цепочка: <название>",
            "- [ГИПОТЕЗА] первопричина: <кратко>",
            "- Почему это вероятно: <ссылка на timestamp/логи/метрики>",
            "- Что проверить для подтверждения/опровержения: <конкретные действия/данные>",
            "Если инцидент один — блок всё равно обязателен и должен содержать минимум 2 гипотезы.",
        ]
    )
    if not base:
        return f"{chain_block}\n\n{incident_link_block}\n\n{root_cause_hypotheses_block}"
    return (
        f"{base}\n\n{chain_block}\n\n{incident_link_block}\n\n{root_cause_hypotheses_block}"
    )


def _read_prompt_setting(name: str) -> str:
    return str(getattr(settings, name, "") or "").strip()


def _resolve_anti_hallucination_rules() -> str:
    custom = _read_prompt_setting("CONTROL_PLANE_LLM_ANTI_HALLUCINATION_RULES")
    return custom or DEFAULT_ANTI_HALLUCINATION_RULES


def _render_prompt_template(template: str, values: Dict[str, Any]) -> str:
    rendered = str(template)
    # Basic Handlebars compatibility for user-provided templates.
    rendered = re.sub(
        r"\{\{#each\s+map_summaries\}\}[\s\S]*?\{\{\/each\}\}",
        "{summaries_text}",
        rendered,
        flags=re.IGNORECASE,
    )
    rendered = re.sub(
        r"\{\{#each\s+source_summaries\}\}[\s\S]*?\{\{\/each\}\}",
        "{source_summaries_text}",
        rendered,
        flags=re.IGNORECASE,
    )

    def _replace_if_data_type(match: re.Match[str]) -> str:
        body = match.group(1)
        return body if str(values.get("data_type", "")).lower() == "aggregated" else ""

    rendered = re.sub(
        r"\{\{#if\s+data_type\s*==\s*\"aggregated\"\s*\}\}([\s\S]*?)\{\{\/if\}\}",
        _replace_if_data_type,
        rendered,
        flags=re.IGNORECASE,
    )

    def _replace_if_key(match: re.Match[str]) -> str:
        key = str(match.group(1) or "").strip()
        body = match.group(2)
        value = values.get(key)
        return body if bool(value) else ""

    rendered = re.sub(
        r"\{\{#if\s+([a-zA-Z0-9_]+)\s*\}\}([\s\S]*?)\{\{\/if\}\}",
        _replace_if_key,
        rendered,
        flags=re.IGNORECASE,
    )

    def _replace_var(match: re.Match[str]) -> str:
        key = str(match.group(1) or "").strip()
        if key.startswith("this."):
            key = key.split(".", 1)[1]
        return "{" + key + "}"

    rendered = re.sub(
        r"\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}",
        _replace_var,
        rendered,
    )
    safe_values = _SafeFormatDict({k: "" if v is None else str(v) for k, v in values.items()})
    return rendered.format_map(safe_values)


def _ctx_value(ctx: Optional[Dict[str, Any]], key: str, default: Any = "") -> str:
    if not isinstance(ctx, dict):
        return "" if default is None else str(default)
    value = ctx.get(key, default)
    if value is None:
        return ""
    return str(value)


def _validate_iso_timestamp(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp must not be empty")
    datetime.fromisoformat(text.replace("Z", "+00:00"))
    return text


class MapEventSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MapEvidenceType(str, Enum):
    FACT = "FACT"
    HYPOTHESIS = "HYPOTHESIS"


class MapAlertStatus(str, Enum):
    EXPLAINED = "EXPLAINED"
    PARTIALLY = "PARTIALLY"
    NOT_EXPLAINED = "NOT_EXPLAINED"
    NOT_SEEN_IN_BATCH = "NOT_SEEN_IN_BATCH"


class MapHypothesisStatus(str, Enum):
    ACTIVE = "active"
    MERGED = "merged"
    CONFLICTING = "conflicting"
    DISMISSED = "dismissed"


class MapRecommendationPriority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class MapContextModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    batch_id: str
    time_range_start: str
    time_range_end: str
    total_log_entries: int = Field(ge=0)
    source_query: List[str]
    source_services: List[str] = Field(default_factory=list)

    @field_validator("time_range_start", "time_range_end")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        return _validate_iso_timestamp(value)


class MapTimelineEventModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    timestamp: str
    source: str
    description: str
    severity: MapEventSeverity
    importance: float = Field(ge=0.0, le=1.0)
    evidence_type: MapEvidenceType
    evidence_quote: str = ""
    tags: List[str] = Field(default_factory=list)

    @field_validator("timestamp")
    @classmethod
    def _validate_ts(cls, value: str) -> str:
        return _validate_iso_timestamp(value)

    @field_validator("evidence_type", mode="before")
    @classmethod
    def _normalize_evidence_type(cls, value: Any) -> Any:
        raw = str(value or "").strip().upper()
        if raw in {"ФАКТ", "FACT"}:
            return "FACT"
        if raw in {"ГИПОТЕЗА", "HYPOTHESIS"}:
            return "HYPOTHESIS"
        return value


class MapCausalLinkModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    cause_event_id: str
    effect_event_id: str
    mechanism: str
    confidence: float = Field(ge=0.0, le=1.0)


class MapAlertRefModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    alert_id: str
    status: MapAlertStatus
    related_events: List[str] = Field(default_factory=list)
    explanation: str = ""


class MapHypothesisModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    related_alert_ids: List[str] = Field(default_factory=list)
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_events: List[str] = Field(default_factory=list)
    contradicting_events: List[str] = Field(default_factory=list)
    status: MapHypothesisStatus = MapHypothesisStatus.ACTIVE


class MapPinnedFactModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    fact: str
    evidence_quote: str = ""
    relevance: str = ""
    importance: float = Field(ge=0.0, le=1.0)


class MapGapModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    description: str
    between_events: List[str] = Field(default_factory=list)
    missing_data: str = ""


class MapImpactModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    affected_services: List[str] = Field(default_factory=list)
    affected_operations: List[str] = Field(default_factory=list)
    error_counts: List[str] = Field(default_factory=list)
    degradation_period_start: str = ""
    degradation_period_end: str = ""

    @field_validator("degradation_period_start", "degradation_period_end")
    @classmethod
    def _validate_degradation_period(cls, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        return _validate_iso_timestamp(raw)


class MapConflictSideModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str
    supporting_events: List[str] = Field(default_factory=list)


class MapConflictModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    description: str
    side_a: MapConflictSideModel
    side_b: MapConflictSideModel
    resolution: str = ""


class MapGapPeriodModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _validate_period_ts(cls, value: str) -> str:
        return _validate_iso_timestamp(value)


class MapDataQualityModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    is_empty: bool
    noise_ratio: float = Field(ge=0.0, le=1.0)
    has_gaps: bool = False
    gap_periods: List[MapGapPeriodModel] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _sync_gap_flags(self) -> "MapDataQualityModel":
        if self.gap_periods and not self.has_gaps:
            self.has_gaps = True
        if not self.gap_periods and self.has_gaps:
            self.has_gaps = False
        return self


class MapRecommendationModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    priority: MapRecommendationPriority
    action: str
    rationale: str
    related_hypothesis_ids: List[str] = Field(default_factory=list)


class MapBatchSummaryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    context: MapContextModel
    timeline: List[MapTimelineEventModel] = Field(default_factory=list)
    causal_links: List[MapCausalLinkModel] = Field(default_factory=list)
    alert_refs: List[MapAlertRefModel] = Field(default_factory=list)
    hypotheses: List[MapHypothesisModel] = Field(default_factory=list)
    pinned_facts: List[MapPinnedFactModel] = Field(default_factory=list)
    gaps: List[MapGapModel] = Field(default_factory=list)
    impact: MapImpactModel = Field(default_factory=MapImpactModel)
    conflicts: List[MapConflictModel] = Field(default_factory=list)
    data_quality: MapDataQualityModel
    preliminary_recommendations: List[MapRecommendationModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_references(self) -> "MapBatchSummaryModel":
        event_ids = {item.id for item in self.timeline}
        hypothesis_ids = {item.id for item in self.hypotheses}
        for link in self.causal_links:
            if link.cause_event_id not in event_ids:
                raise ValueError(f"cause_event_id `{link.cause_event_id}` not found in timeline")
            if link.effect_event_id not in event_ids:
                raise ValueError(f"effect_event_id `{link.effect_event_id}` not found in timeline")
        for alert in self.alert_refs:
            for event_id in alert.related_events:
                if event_id not in event_ids:
                    raise ValueError(f"alert `{alert.alert_id}` references unknown event `{event_id}`")
        for hypothesis in self.hypotheses:
            for event_id in hypothesis.supporting_events:
                if event_id not in event_ids:
                    raise ValueError(
                        f"hypothesis `{hypothesis.id}` references unknown supporting event `{event_id}`"
                    )
            for event_id in hypothesis.contradicting_events:
                if event_id not in event_ids:
                    raise ValueError(
                        f"hypothesis `{hypothesis.id}` references unknown contradicting event `{event_id}`"
                    )
        for gap in self.gaps:
            for event_id in gap.between_events:
                if event_id not in event_ids:
                    raise ValueError(f"gap `{gap.id}` references unknown event `{event_id}`")
        for conflict in self.conflicts:
            for event_id in conflict.side_a.supporting_events + conflict.side_b.supporting_events:
                if event_id not in event_ids:
                    raise ValueError(f"conflict `{conflict.id}` references unknown event `{event_id}`")
        for recommendation in self.preliminary_recommendations:
            for hypothesis_id in recommendation.related_hypothesis_ids:
                if hypothesis_id not in hypothesis_ids:
                    raise ValueError(
                        f"recommendation `{recommendation.id}` references unknown hypothesis `{hypothesis_id}`"
                    )
        return self


class InstructorFinalReportSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    text: str


class InstructorFinalReports(BaseModel):
    model_config = ConfigDict(extra="ignore")
    structured_sections: List[InstructorFinalReportSection] = Field(default_factory=list)
    freeform_sections: List[InstructorFinalReportSection] = Field(default_factory=list)
    structured_report: str = ""
    freeform_report: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "InstructorFinalReports":
        if not str(self.structured_report or "").strip() and not self.structured_sections:
            raise ValueError("structured report is empty")
        if not str(self.freeform_report or "").strip() and not self.freeform_sections:
            raise ValueError("freeform report is empty")
        return self


def _normalize_summary_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def _preview_text(value: Any, max_chars: int = 1200) -> str:
    text = str(value or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _payload_for_progress_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in (payload or {}).items():
        if key == "batch_logs" and isinstance(value, list):
            safe["batch_logs_count"] = len(value)
            if value:
                try:
                    safe["batch_logs_sample"] = {
                        k: _preview_text(v, max_chars=300)
                        for k, v in dict(value[0]).items()
                    }
                except Exception:
                    safe["batch_logs_sample"] = "<unavailable>"
            continue
        if isinstance(value, str):
            safe[key] = _preview_text(value, max_chars=1200)
        elif isinstance(value, list):
            safe[key] = f"<list:{len(value)}>"
        elif isinstance(value, dict):
            safe[key] = "<dict>"
        else:
            safe[key] = value
    return safe


def has_required_env() -> bool:
    return bool(str(settings.OPENAI_API_BASE_DB).strip()) and bool(str(settings.OPENAI_API_KEY_DB).strip())


def _build_chat_completions_url(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _build_openai_base_url(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _default_llm_system_prompt() -> str:
    return (
        "Ты — senior SRE-аналитик инцидентов. Анализируй логи и метрики строго на основе данных.\n"
        "Принципы:\n"
        "1) Только факты: каждое утверждение подтверждай timestamp/сообщением/значением.\n"
        "2) Маркировка: [ФАКТ] — прямое подтверждение, [ГИПОТЕЗА] — предположение.\n"
        "3) Если данных недостаточно — пиши \"данных недостаточно\", не додумывай.\n"
        "4) Хронология обязательна: строй цепочку событий по времени.\n"
        "5) Главный результат: причинно-следственные цепочки (триггер→распространение→последствия→алерты).\n"
        "6) Если звенья не связаны данными — отмечай разрывы цепочки и нужные данные для закрытия.\n"
        "7) Возможны несколько независимых цепочек/инцидентов: не склеивай их без механизма связи.\n"
        "8) Отделяй [РЕЛЕВАНТНО] события от [ФОН]/[НЕЯСНО].\n"
        "9) Для агрегированных логов: строка=группа событий, cnt=масштаб, argMin-поля=пример.\n"
        "10) Корреляция по времени не равна причинности."
    )


def _resolve_llm_system_prompt() -> str:
    custom_system_prompt = str(getattr(settings, "CONTROL_PLANE_LLM_SYSTEM_PROMPT", "")).strip()
    return custom_system_prompt or _default_llm_system_prompt()


def _response_body_text(response: Any) -> str:
    try:
        text = getattr(response, "text", None)
        if text is None:
            content = getattr(response, "content", b"")
            if isinstance(content, (bytes, bytearray)):
                text = bytes(content).decode("utf-8", errors="replace")
            else:
                text = str(content)
        return str(text or "")
    except Exception as exc:  # noqa: BLE001
        return f"<response body unavailable: {exc}>"


def _raise_for_status_with_body(response: Any) -> None:
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = getattr(response, "status_code", None)
        url = getattr(response, "url", None)
        body_text = _response_body_text(response)
        logger.error(
            "LLM HTTP error status=%s url=%s response_body=%s",
            status,
            url or "<unknown>",
            body_text,
        )
        enriched = str(exc)
        if body_text:
            enriched = f"{enriched}\nRESPONSE_BODY:\n{body_text}"
        raise requests.exceptions.HTTPError(enriched, response=response) from exc


def _extract_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is None and "value" in item:
                    text = item.get("value")
                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _parse_chat_completion_response(data: Any) -> Tuple[str, str]:
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            finish_reason = str(choice.get("finish_reason") or "").strip().lower()
            message = choice.get("message", {})
            content = ""
            if isinstance(message, dict):
                content = _extract_message_content(message.get("content"))
            if not content and "text" in choice:
                content = _extract_message_content(choice.get("text"))
            if content:
                return content, finish_reason
        output_text = data.get("output_text")
        if output_text is not None:
            return _extract_message_content(output_text), ""
    return str(data), ""


def communicate_with_llm(message: str, system_prompt: str = "", timeout: float = 600.0) -> str:
    if not has_required_env():
        raise RuntimeError("OPENAI_API_BASE_DB and OPENAI_API_KEY_DB are required")

    url = _build_chat_completions_url(str(settings.OPENAI_API_BASE_DB))
    headers = {
        "Authorization": f"Bearer {str(settings.OPENAI_API_KEY_DB)}",
        "Content-Type": "application/json",
    }
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    base_payload: Dict[str, Any] = {
        "model": str(settings.LLM_MODEL_ID),
        "temperature": 0.1,
    }
    configured_max_tokens = int(getattr(settings, "CONTROL_PLANE_LLM_MAX_TOKENS", 0) or 0)
    if configured_max_tokens > 0:
        base_payload["max_tokens"] = configured_max_tokens
    continue_on_length = bool(
        getattr(settings, "CONTROL_PLANE_LLM_CONTINUE_ON_LENGTH", True)
    )
    continue_round_limit = max(
        int(getattr(settings, "CONTROL_PLANE_LLM_CONTINUE_MAX_ROUNDS", 12) or 12),
        1,
    )

    dialog_messages: List[Dict[str, str]] = list(messages)
    collected_parts: List[str] = []
    for round_idx in range(1, continue_round_limit + 1):
        payload = dict(base_payload)
        payload["messages"] = list(dialog_messages)
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if (
            response.status_code == 400
            and "max_tokens" in payload
            and configured_max_tokens > 0
        ):
            # Some OpenAI-compatible gateways reject max_tokens or specific values.
            # Retry once without explicit max_tokens before surfacing the error.
            logger.warning(
                "LLM gateway returned 400 with max_tokens=%s; retrying the same call without max_tokens.",
                configured_max_tokens,
            )
            logger.warning(
                "LLM 400 response body (with max_tokens): %s",
                _response_body_text(response),
            )
            payload_without_max = dict(payload)
            payload_without_max.pop("max_tokens", None)
            response = requests.post(
                url,
                json=payload_without_max,
                headers=headers,
                timeout=timeout,
            )
        _raise_for_status_with_body(response)
        data = response.json()
        chunk_text, finish_reason = _parse_chat_completion_response(data)
        collected_parts.append(str(chunk_text or ""))

        can_continue = continue_on_length and finish_reason in {"length", "max_tokens"}
        if not can_continue:
            break
        if round_idx >= continue_round_limit:
            logger.warning(
                "LLM output still marked as truncated after %s continuation rounds; "
                "returning accumulated text as-is.",
                continue_round_limit,
            )
            break

        logger.warning(
            "LLM finish_reason=%s: requesting continuation chunk %s/%s",
            finish_reason,
            round_idx + 1,
            continue_round_limit,
        )
        dialog_messages.append({"role": "assistant", "content": str(chunk_text or "")})
        dialog_messages.append(
            {
                "role": "user",
                "content": (
                    "Продолжи строго с того места, где остановился. "
                    "Не повторяй предыдущие абзацы, сохрани формат ответа."
                ),
            }
        )

    return "".join(collected_parts).strip()


def _normalize_period(
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> Tuple[str, str]:
    if period_start and period_end:
        return period_start, period_end
    if start_dt is not None and end_dt is not None:
        return start_dt.isoformat(), end_dt.isoformat()
    raise ValueError("Provide either period_start+period_end or start_dt+end_dt")


def _extract_batch_period(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    timestamps: List[pd.Timestamp] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_ts = None
        for key in ("timestamp", "start_time", "end_time", "ts", "time", "datetime"):
            if row.get(key) is not None:
                raw_ts = row.get(key)
                break
        if raw_ts is None:
            continue
        ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        timestamps.append(ts)

    if not timestamps:
        return None, None

    start_ts = min(timestamps).isoformat().replace("+00:00", "Z")
    end_ts = max(timestamps).isoformat().replace("+00:00", "Z")
    return start_ts, end_ts


def _resolve_service(anomaly: Optional[Dict[str, Any]]) -> str:
    if anomaly and anomaly.get("service"):
        return str(anomaly["service"])
    raise ValueError(
        "Missing anomaly['service'] for logs summarization. "
        "Pass service in anomaly payload."
    )


def _resolve_logs_query_template() -> str:
    template = str(settings.CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY).strip()
    if template:
        return template
    raise ValueError(
        "Set CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY in .env "
        "(full SQL string; optionally with placeholders)."
    )


def _resolve_logs_fetch_mode() -> str:
    raw_mode = str(getattr(settings, "CONTROL_PLANE_LOGS_FETCH_MODE", "time_window")).strip().lower()
    aliases = {
        "time_window": "time_window",
        "window": "time_window",
        "lookback": "time_window",
        "tail_n_logs": "tail_n_logs",
        "tail_n": "tail_n_logs",
        "last_n_logs": "tail_n_logs",
    }
    resolved = aliases.get(raw_mode)
    if resolved is None:
        logger.warning(
            "Unknown CONTROL_PLANE_LOGS_FETCH_MODE=%s; fallback to time_window",
            raw_mode,
        )
        return "time_window"
    return resolved


def _strip_trailing_limit_offset(query: str) -> str:
    stripped = query.strip().rstrip(";")
    without_limit_offset = re.sub(
        r"(?is)\s+LIMIT\s+\d+\s+OFFSET\s+\d+\s*$",
        "",
        stripped,
    )
    without_offset_only = re.sub(
        r"(?is)\s+OFFSET\s+\d+\s*$",
        "",
        without_limit_offset,
    )
    without_limit_only = re.sub(
        r"(?is)\s+LIMIT\s+\d+\s*$",
        "",
        without_offset_only,
    )
    return without_limit_only


def _build_tail_paged_query(
    *,
    base_query: str,
    tail_limit: int,
    limit: int,
    offset: int,
) -> str:
    safe_tail_limit = max(int(tail_limit), 1)
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    return (
        "SELECT * FROM ("
        "SELECT * FROM ("
        f"{base_query}"
        f") AS cp_src ORDER BY timestamp DESC LIMIT {safe_tail_limit}"
        ") AS cp_tail "
        "ORDER BY timestamp ASC "
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _wrap_with_limit_offset(
    *,
    base_query: str,
    limit: int,
    offset: int,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    return (
        "SELECT * FROM ("
        f"{base_query}"
        ") AS cp_logs_page "
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _render_logs_query(
    *,
    query_template: str,
    period_start: str,
    period_end: str,
    limit: int,
    offset: int,
    service: str,
    last_ts: Optional[str] = None,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    # last_ts defaults to period_start so keyset queries work correctly on the first page
    effective_last_ts = last_ts if last_ts is not None else period_start
    params = _SafeFormatDict(
        period_start=period_start,
        period_end=period_end,
        start=period_start,
        end=period_end,
        start_iso=period_start,
        end_iso=period_end,
        limit=safe_limit,
        page_limit=safe_limit,
        offset=safe_offset,
        service=service,
        last_ts=effective_last_ts,
    )
    return query_template.strip().rstrip(";").format_map(params)


def _query_logs_df(query: str) -> pd.DataFrame:
    try:
        import clickhouse_connect
    except Exception as exc:
        raise ImportError("Для чтения логов нужен пакет clickhouse-connect") from exc

    host = str(settings.CONTROL_PLANE_LOGS_CLICKHOUSE_HOST).strip()
    if not host:
        raise ValueError(
            "Set CONTROL_PLANE_LOGS_CLICKHOUSE_HOST in .env for logs summarization"
        )

    client = clickhouse_connect.get_client(
        host=host,
        port=int(settings.CONTROL_PLANE_LOGS_CLICKHOUSE_PORT),
        username=str(settings.CONTROL_PLANE_LOGS_CLICKHOUSE_USERNAME).strip() or None,
        password=str(settings.CONTROL_PLANE_LOGS_CLICKHOUSE_PASSWORD).strip() or None,
    )
    try:
        return client.query_df(query)
    finally:
        try:
            client.close()
        except Exception:
            logger.warning("ClickHouse client close failed for logs query")


def _build_db_fetch_page(
    anomaly: Optional[Dict[str, Any]],
    *,
    fetch_mode: str,
    tail_limit: int,
    on_error: Optional[Callable[[str], None]] = None,
) -> Callable[..., List[Dict[str, Any]]]:
    service = _resolve_service(anomaly)
    query_template = _resolve_logs_query_template()
    query_template_lc = query_template.lower()
    has_offset_placeholder = "{offset}" in query_template_lc
    # Keyset pagination: if template contains {last_ts}, we use timestamp-based pagination
    # instead of LIMIT/OFFSET to avoid ClickHouse MEMORY_LIMIT_EXCEEDED on large offsets.
    has_last_ts_placeholder = "{last_ts}" in query_template_lc
    # Mutable single-element list so the inner closure can update state between pages
    _keyset_last_ts: List[Optional[str]] = [None]

    def _db_fetch_page(
        *,
        columns: Sequence[str],
        period_start: str,
        period_end: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        if has_last_ts_placeholder:
            # Keyset mode: ignore `offset`, track last_ts across calls.
            # On the very first call _keyset_last_ts[0] is None, so _render_logs_query
            # defaults last_ts to period_start (= fetch everything from the start).
            rendered_query = _render_logs_query(
                query_template=query_template,
                period_start=period_start,
                period_end=period_end,
                limit=limit,
                offset=0,
                service=service,
                last_ts=_keyset_last_ts[0],
            )
            try:
                page_df = _query_logs_df(rendered_query)
            except Exception as exc:
                msg = (
                    f"ClickHouse keyset query failed "
                    f"(service={service}, last_ts={_keyset_last_ts[0]}): {exc}"
                )
                logger.warning(msg)
                if on_error:
                    on_error(msg)
                return []

            if page_df.empty:
                return []

            # Advance last_ts to the maximum timestamp in this page
            if "timestamp" in page_df.columns:
                max_ts = pd.to_datetime(page_df["timestamp"], utc=True, errors="coerce").max()
                if not pd.isna(max_ts):
                    _keyset_last_ts[0] = max_ts.isoformat()

            records = page_df.to_dict(orient="records")
            if not columns:
                return [dict(row) for row in records]
            projected_rows: List[Dict[str, Any]] = []
            for row in records:
                projected_rows.append({col: row.get(col) for col in columns})
            return projected_rows

        # --- Legacy path ---
        effective_period_start = (
            _EARLIEST_PERIOD_START if fetch_mode == "tail_n_logs" else period_start
        )
        rendered_query = _render_logs_query(
            query_template=query_template,
            period_start=effective_period_start,
            period_end=period_end,
            limit=limit,
            offset=offset,
            service=service,
        )
        if fetch_mode == "tail_n_logs":
            base_query = _strip_trailing_limit_offset(rendered_query)
            query = _build_tail_paged_query(
                base_query=base_query,
                tail_limit=tail_limit,
                limit=limit,
                offset=offset,
            )
        elif has_offset_placeholder:
            # SQL template already supports paging placeholders.
            query = rendered_query
        else:
            # Auto-paging fallback:
            # if template has no OFFSET placeholder, page it externally to avoid
            # "first 1000 rows only" behavior when SQL has a fixed LIMIT.
            base_query = _strip_trailing_limit_offset(rendered_query)
            query = _wrap_with_limit_offset(
                base_query=base_query,
                limit=limit,
                offset=offset,
            )

        try:
            page_df = _query_logs_df(query)
        except Exception as exc:
            msg = f"ClickHouse query failed (service={service}, offset={offset}): {exc}"
            logger.warning(msg)
            if on_error:
                on_error(msg)
            return []

        if page_df.empty:
            return []
        records = page_df.to_dict(orient="records")
        if not columns:
            return [dict(row) for row in records]
        projected_rows: List[Dict[str, Any]] = []
        for row in records:
            projected_rows.append({col: row.get(col) for col in columns})
        return projected_rows

    return _db_fetch_page


def _build_count_query(data_query: str) -> str:
    without_limit_only = _strip_trailing_limit_offset(data_query)
    return f"SELECT count() AS total_rows FROM ({without_limit_only}) AS cp_logs"


def _estimate_total_logs(
    *,
    anomaly: Optional[Dict[str, Any]],
    period_start: str,
    period_end: str,
    page_limit: int,
    fetch_mode: str,
    tail_limit: int,
) -> Optional[int]:
    try:
        service = _resolve_service(anomaly)
        query_template = _resolve_logs_query_template()
        sample_query = _render_logs_query(
            query_template=query_template,
            period_start=(
                _EARLIEST_PERIOD_START if fetch_mode == "tail_n_logs" else period_start
            ),
            period_end=period_end,
            limit=page_limit,
            offset=0,
            service=service,
            last_ts=period_start,  # keyset: count from period start
        )
        count_query = _build_count_query(sample_query)
        df = _query_logs_df(count_query)
        if df.empty:
            return None
        if "total_rows" in df.columns:
            total_rows = int(df.iloc[0]["total_rows"])
        else:
            total_rows = int(df.iloc[0, 0])
        if fetch_mode == "tail_n_logs":
            return min(total_rows, max(int(tail_limit), 1))
        return total_rows
    except Exception as exc:
        logger.warning("Failed to estimate total logs for progress: %s", exc)
        return None


def _heuristic_llm_call(prompt: str, error: Optional[str] = None) -> str:
    error_line = f"ОШИБКА: {error}" if error else "LLM не настроена (OPENAI_API_BASE_DB / OPENAI_API_KEY_DB)."
    return (
        "[LLM ERROR]\n\n"
        f"{error_line}\n\n"
        "Summary по этому шагу не получен из LLM.\n"
        "Проверь параметры запроса/размер батча/таймаут и перезапусти шаг."
    )


def _is_llm_error_stub(summary_text: str) -> bool:
    raw = str(summary_text or "").strip().lower()
    if not raw:
        return False
    return (
        raw.startswith("[llm error]")
        or raw.startswith("[llm недоступна")
        or "эвристический fallback" in raw
    )


def _is_read_timeout_exception(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return True
    exc_type_name = type(exc).__name__.lower()
    if "timeout" in exc_type_name and "read" in exc_type_name:
        return True
    # OpenAI Python client timeout class
    if "apitimeouterror" in exc_type_name:
        return True
    text = str(exc).strip().lower()
    return (
        "read timed out" in text
        or "read timeout" in text
        or "readtimeout" in text
    )


def _is_non_retryable_llm_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if 400 <= status_code < 500 and status_code not in (408, 409, 429):
            return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            # Most 4xx errors are request-shape/auth/data problems and should not
            # be retried in a loop. Keep 408/409/429 retryable.
            if 400 <= status_code < 500 and status_code not in (408, 409, 429):
                return True
    text = str(exc).strip().lower()
    if "error code: 400" in text and "badrequesterror" in text:
        return True
    if "400 client error" in text and "bad request" in text:
        return True
    return False


def _is_400_bad_request_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code == 400:
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and status_code == 400:
            return True
    text = str(exc).strip().lower()
    return (
        ("400 client error" in text and "bad request" in text)
        or ("error code: 400" in text and "badrequesterror" in text)
    )


def _is_invalid_grammar_request_exception(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    return (
        "invalid grammar request" in text
        or "hosted_vllmexception - invalid grammar request" in text
        or "grammar request with cache hit" in text
    )


def _make_llm_call(
    max_retries: int = -1,
    retry_delay: float = 10.0,
    on_retry: Optional[Callable[[int, int, Exception], None]] = None,
    on_attempt: Optional[Callable[[int, int, float], None]] = None,
    on_result: Optional[Callable[[int, int, bool, float, Optional[str]], None]] = None,
    llm_timeout: float = 600.0,
) -> LLMTextCaller:
    if not has_required_env():
        raise RuntimeError(
            "OPENAI_API_BASE_DB и OPENAI_API_KEY_DB обязательны для LLM-суммаризации."
        )

    _system_prompt = _resolve_llm_system_prompt()

    def _llm_call(prompt: str) -> str:
        last_exc: Optional[Exception] = None
        retries = int(max_retries)
        infinite_retries = retries < 0
        configured_total_attempts = -1 if infinite_retries else (max(retries, 0) + 1)
        effective_total_attempts = configured_total_attempts
        base_timeout = max(float(llm_timeout), 1.0)
        current_timeout = base_timeout
        attempt_no = 0
        while True:
            attempt_no += 1
            if on_attempt is not None:
                try:
                    on_attempt(attempt_no, effective_total_attempts, current_timeout)
                except Exception:
                    pass
            started = time.monotonic()
            try:
                response = communicate_with_llm(
                    message=prompt,
                    system_prompt=_system_prompt,
                    timeout=current_timeout,
                )
                elapsed = max(time.monotonic() - started, 0.0)
                if on_result is not None:
                    try:
                        on_result(attempt_no, effective_total_attempts, True, elapsed, None)
                    except Exception:
                        pass
                return str(response).strip()
            except Exception as exc:
                last_exc = exc
                elapsed = max(time.monotonic() - started, 0.0)
                is_read_timeout = _is_read_timeout_exception(exc)
                is_non_retryable = _is_non_retryable_llm_exception(exc)
                if is_read_timeout:
                    # For ReadTimeout we keep waiting indefinitely and progressively
                    # increase the request timeout on each occurrence.
                    effective_total_attempts = -1
                result_total_attempts = (
                    attempt_no if is_non_retryable else effective_total_attempts
                )
                if on_result is not None:
                    try:
                        on_result(
                            attempt_no,
                            result_total_attempts,
                            False,
                            elapsed,
                            str(exc),
                        )
                    except Exception:
                        pass
                if is_non_retryable:
                    logger.error(
                        "LLM non-retryable error (stop retries): %s",
                        exc,
                    )
                    raise
                can_retry = (
                    is_read_timeout
                    or infinite_retries
                    or attempt_no <= max(retries, 0)
                )
                if can_retry:
                    if is_read_timeout:
                        next_timeout = current_timeout + base_timeout
                        logger.warning(
                            "LLM ReadTimeout on attempt %d; next timeout %.1fs (prev %.1fs)",
                            attempt_no,
                            next_timeout,
                            current_timeout,
                        )
                        current_timeout = next_timeout
                    elif infinite_retries:
                        logger.warning(
                            "LLM retry %d/∞ after error: %s",
                            attempt_no + 1,
                            exc,
                        )
                    else:
                        logger.warning(
                            "LLM retry %d/%d after error: %s",
                            attempt_no + 1,
                            configured_total_attempts,
                            exc,
                        )
                    if on_retry is not None:
                        try:
                            on_retry(attempt_no, effective_total_attempts, exc)
                        except Exception:
                            pass
                    # Fixed wait between retries (no exponential backoff).
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                else:
                    logger.exception(
                        "LLM все %d попытки исчерпаны; поднимаем ошибку",
                        configured_total_attempts,
                    )
                    break
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM call failed without explicit exception")

    return _llm_call


class PeriodLogSummarizer:
    PROBLEM_KEYWORDS = (
        "error",
        "exception",
        "timeout",
        "failed",
        "fail",
        "fatal",
        "critical",
        "panic",
        "denied",
        "refused",
        "unavailable",
    )

    def __init__(
        self,
        *,
        db_fetch_page: DBPageFetcher,
        llm_call: LLMTextCaller,
        config: SummarizerConfig | None = None,
        on_progress: Optional[ProgressCallback] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.db_fetch_page = db_fetch_page
        self.llm_call = llm_call
        self.config = config or SummarizerConfig()
        self.on_progress = on_progress
        self.prompt_context = prompt_context or {}
        self._instructor_client_cache: Dict[str, Any] = {}
        self._instructor_force_json_mode: bool = False
        # Some gateways reject JSON grammar for large schemas (MAP model).
        # If that happens once, MAP switches to legacy JSON text parsing for this run.
        self._instructor_map_disabled_due_grammar: bool = False

    def _instructor_enabled(self) -> bool:
        return (
            bool(getattr(self.config, "use_instructor", True))
            and bool(instructor is not None)
            and bool(OpenAI is not None)
            and has_required_env()
        )

    def _instructor_tool_calling_enabled(self) -> bool:
        if self._instructor_force_json_mode:
            return False
        cfg_value = getattr(self.config, "model_supports_tool_calling", None)
        if cfg_value is None:
            return bool(getattr(settings, "CONTROL_PLANE_LLM_SUPPORTS_TOOL_CALLING", True))
        return bool(cfg_value)

    def _resolve_instructor_mode(self) -> Any:
        mode_enum = getattr(instructor, "Mode", None)
        if mode_enum is None:
            return None
        if self._instructor_tool_calling_enabled() and hasattr(mode_enum, "TOOLS"):
            return getattr(mode_enum, "TOOLS")
        if hasattr(mode_enum, "JSON"):
            return getattr(mode_enum, "JSON")
        return None

    def _get_instructor_client(self) -> Any:
        if not self._instructor_enabled():
            raise RuntimeError("Instructor runtime is unavailable for structured calls.")
        mode = self._resolve_instructor_mode()
        cache_key = "|".join(
            [
                str(_build_openai_base_url(str(settings.OPENAI_API_BASE_DB))),
                str(settings.LLM_MODEL_ID),
                str(mode),
            ]
        )
        cached = self._instructor_client_cache.get(cache_key)
        if cached is not None:
            return cached
        base_url = _build_openai_base_url(str(settings.OPENAI_API_BASE_DB))
        api_key = str(settings.OPENAI_API_KEY_DB)
        openai_client = OpenAI(base_url=base_url, api_key=api_key)
        if mode is not None:
            client = instructor.from_openai(openai_client, mode=mode)
        else:
            client = instructor.from_openai(openai_client)
        self._instructor_client_cache[cache_key] = client
        return client

    def _tool_calling_parser_error(self, exc: Exception) -> bool:
        text = str(exc or "").lower()
        return "tool_choice" in text and "tool-call-parser" in text

    def _structured_call_timeout_base(self) -> float:
        prompt_timeout = self.prompt_context.get("llm_timeout")
        timeout_value = pd.to_numeric(prompt_timeout, errors="coerce")
        if pd.isna(timeout_value):
            timeout_value = pd.to_numeric(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_TIMEOUT", 600),
                errors="coerce",
            )
        try:
            return max(float(timeout_value), 1.0)
        except Exception:
            return 600.0

    def _structured_call_retry_limit(self) -> int:
        prompt_retries = self.prompt_context.get("llm_max_retries")
        retries_value = pd.to_numeric(prompt_retries, errors="coerce")
        if pd.isna(retries_value):
            retries_value = pd.to_numeric(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_RETRIES", -1),
                errors="coerce",
            )
        try:
            return int(retries_value)
        except Exception:
            return -1

    def _call_structured_with_instructor(
        self,
        *,
        prompt: str,
        response_model: Type[TModel],
        stage: str,
    ) -> Tuple[TModel, int]:
        if not self._instructor_enabled():
            raise RuntimeError("Instructor is disabled or unavailable.")

        retries = self._structured_call_retry_limit()
        infinite_retries = retries < 0
        configured_total_attempts = -1 if infinite_retries else (max(retries, 0) + 1)
        base_timeout = self._structured_call_timeout_base()
        current_timeout = base_timeout
        attempt_no = 0
        last_exc: Optional[Exception] = None
        while True:
            attempt_no += 1
            mode_label = "TOOLS" if self._instructor_tool_calling_enabled() else "JSON"
            logger.info(
                "Instructor structured call start | stage=%s | model=%s | mode=%s | attempt=%s/%s | timeout=%.1fs",
                stage,
                str(settings.LLM_MODEL_ID),
                mode_label,
                attempt_no,
                "∞" if configured_total_attempts < 0 else configured_total_attempts,
                current_timeout,
            )
            started = time.monotonic()
            try:
                client = self._get_instructor_client()
                result = client.chat.completions.create(
                    model=str(settings.LLM_MODEL_ID),
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": _resolve_llm_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_retries=0,
                    timeout=current_timeout,
                )
                elapsed = max(time.monotonic() - started, 0.0)
                logger.info(
                    "Instructor structured call done | stage=%s | attempt=%s | elapsed=%.2fs",
                    stage,
                    attempt_no,
                    elapsed,
                )
                return result, attempt_no
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                elapsed = max(time.monotonic() - started, 0.0)
                logger.warning(
                    "Instructor structured call failed | stage=%s | attempt=%s | elapsed=%.2fs | error=%s",
                    stage,
                    attempt_no,
                    elapsed,
                    exc,
                )
                if self._tool_calling_parser_error(exc) and self._instructor_tool_calling_enabled():
                    logger.warning(
                        "Instructor mode TOOLS is unsupported by current gateway. "
                        "Switching to JSON mode for subsequent structured calls."
                    )
                    self._instructor_force_json_mode = True
                    continue
                if (
                    _is_invalid_grammar_request_exception(exc)
                    and self._instructor_tool_calling_enabled()
                ):
                    logger.warning(
                        "Instructor structured call got invalid grammar in TOOLS mode. "
                        "Switching to JSON mode and retrying."
                    )
                    self._instructor_force_json_mode = True
                    continue
                is_read_timeout = _is_read_timeout_exception(exc)
                is_non_retryable = _is_non_retryable_llm_exception(exc)
                if is_non_retryable:
                    raise
                can_retry = is_read_timeout or infinite_retries or attempt_no <= max(retries, 0)
                if not can_retry:
                    break
                if is_read_timeout:
                    next_timeout = current_timeout + base_timeout
                    logger.warning(
                        "Instructor ReadTimeout | stage=%s | attempt=%s | timeout %.1fs -> %.1fs",
                        stage,
                        attempt_no,
                        current_timeout,
                        next_timeout,
                    )
                    current_timeout = next_timeout
                retry_delay = 10.0
                if retry_delay > 0:
                    time.sleep(retry_delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Instructor structured call failed without explicit exception")

    def _render_rows_as_text(
        self,
        *,
        rows: Sequence[Dict[str, Any]],
        columns: Sequence[str],
    ) -> str:
        rendered: List[str] = []
        for idx, row in enumerate(rows, start=1):
            parts: List[str] = []
            for col in columns:
                if col not in row:
                    continue
                value = row.get(col)
                if value is None or value == "":
                    continue
                parts.append(f"{col}={value}")
            if parts:
                rendered.append(f"{idx}. " + " | ".join(parts))
        return "\n".join(rendered)

    def _split_rows_for_map(
        self,
        *,
        rows: List[Dict[str, Any]],
        columns: Sequence[str],
    ) -> List[List[Dict[str, Any]]]:
        if not rows:
            return []
        # New algorithm works without local token estimation and does not pre-split
        # batches by estimated token size. Split-on-overflow is handled only after
        # actual LLM 400/timeouts.
        return [rows]

    @staticmethod
    def _normalize_incident_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(payload or {})
        impact = data.get("impact")
        if isinstance(impact, dict):
            if "degradation_period" not in impact:
                start = str(impact.get("degradation_period_start") or "").strip()
                end = str(impact.get("degradation_period_end") or "").strip()
                if start and end:
                    impact["degradation_period"] = {"start": start, "end": end}
            impact.pop("degradation_period_start", None)
            impact.pop("degradation_period_end", None)
            data["impact"] = impact
        return data

    def _parse_incident_summary_text(
        self,
        text: str,
        *,
        fallback_batch_id: str,
        fallback_start: str,
        fallback_end: str,
    ) -> Optional[IncidentSummary]:
        payload = self._extract_json_payload(text)
        if not isinstance(payload, dict):
            return None
        normalized = self._normalize_incident_payload(payload)
        context = normalized.get("context")
        if not isinstance(context, dict):
            context = {}
        context.setdefault("batch_id", fallback_batch_id)
        context.setdefault("time_range_start", fallback_start)
        context.setdefault("time_range_end", fallback_end)
        context.setdefault("total_log_entries", 0)
        context.setdefault("source_query", [])
        context.setdefault("source_services", [])
        normalized["context"] = context
        try:
            return IncidentSummary.model_validate(normalized)
        except ValidationError:
            return None

    @staticmethod
    def _incident_summary_to_json(summary: IncidentSummary) -> str:
        return json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2)

    def _build_reduce_prompt_with_premerged_alerts(
        self,
        *,
        period_start: str,
        period_end: str,
        reduce_round: int,
        summaries: List[str],
        premerged_alert_refs: Optional[List[IncidentAlertRef]] = None,
        sources: Optional[List[str]] = None,
    ) -> str:
        base_prompt = self._build_reduce_prompt(
            period_start=period_start,
            period_end=period_end,
            reduce_round=reduce_round,
            summaries=summaries,
            sources=sources,
        )
        if not premerged_alert_refs:
            return base_prompt
        refs_payload = [
            ref.model_dump(mode="json")
            for ref in premerged_alert_refs
        ]
        refs_text = json.dumps(refs_payload, ensure_ascii=False, indent=2)
        return (
            f"{base_prompt}\n\n"
            "PROGRAMMATICALLY MERGED ALERT_REFS (final statuses, do not recalculate):\n"
            f"{refs_text}"
        )

    @staticmethod
    def _summary_is_empty(summary: IncidentSummary) -> bool:
        is_empty_flag = bool(getattr(summary.data_quality, "is_empty", False))
        return is_empty_flag or len(summary.timeline) == 0

    def _group_structured_summaries_by_budget(
        self,
        summaries: List[IncidentSummary],
    ) -> List[List[IncidentSummary]]:
        if not summaries:
            return []
        # Updated algorithm: fixed-size grouping, no local token budgeting.
        group_size = max(int(getattr(self.config, "reduce_group_size", 3) or 3), 1)
        return [summaries[i : i + group_size] for i in range(0, len(summaries), group_size)]

    def _compress_structured_summary_if_needed(
        self,
        summary: IncidentSummary,
        *,
        period_start: str,
        period_end: str,
    ) -> tuple[IncidentSummary, int]:
        # No local token estimation in the updated algorithm.
        # Compression is applied reactively on real 400/timeouts in reduce.
        _ = (period_start, period_end)
        return summary, 0

    def _compress_summary_on_overflow(
        self,
        summary: IncidentSummary,
        *,
        importance_thresholds: Sequence[float] = (0.7, 0.85),
    ) -> tuple[IncidentSummary, int]:
        current = summary
        llm_calls = 0
        target_pct = max(min(int(getattr(self.config, "compression_target_pct", 50) or 50), 90), 20)
        for threshold in importance_thresholds:
            system_text = incident_prompts.COMPRESSION_SYSTEM_PROMPT.format(
                target_pct=target_pct,
                importance_threshold=float(threshold),
            )
            user_text = incident_prompts.COMPRESSION_USER_PROMPT.format(
                summary_json=self._incident_summary_to_json(current),
            )
            prompt = f"{system_text}\n\n{user_text}"
            if self._instructor_enabled():
                try:
                    parsed, attempts = self._call_structured_with_instructor(
                        prompt=prompt,
                        response_model=IncidentSummary,
                        stage="compression",
                    )
                    llm_calls += max(attempts, 1)
                    current = parsed
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Compression structured call failed, fallback to legacy parsing: %s", exc)
            raw = str(self.llm_call(prompt) or "").strip()
            llm_calls += 1
            parsed = self._parse_incident_summary_text(
                raw,
                fallback_batch_id=current.context.batch_id,
                fallback_start=current.context.time_range_start,
                fallback_end=current.context.time_range_end,
            )
            if parsed is None:
                break
            current = parsed
        return current, llm_calls

    @staticmethod
    def _deterministic_merge_alert_refs(
        summaries: List[IncidentSummary],
    ) -> List[IncidentAlertRef]:
        merged = merge_alert_refs(summaries)
        normalized: List[IncidentAlertRef] = []
        for item in merged:
            normalized.append(
                IncidentAlertRef(
                    alert_id=str(item.alert_id),
                    status=item.status,
                    # Keep union of related events from programmatic merge; reduce stage
                    # can remap them to the renumbered timeline ids.
                    related_events=sorted({str(eid) for eid in (item.related_events or []) if str(eid)}),
                    explanation=str(item.explanation or ""),
                )
            )
        return normalized

    def _extract_json_payload(self, raw_text: str) -> Optional[Dict[str, Any]]:
        text = str(raw_text or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
                return payload if isinstance(payload, dict) else None
            except Exception:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                payload = json.loads(candidate)
                return payload if isinstance(payload, dict) else None
            except Exception:
                return None
        return None

    def _derive_source_services(self, rows_chunk: List[Dict[str, Any]]) -> List[str]:
        values: List[str] = []
        for row in rows_chunk:
            for key in ("_source", "service", "source", "host", "node", "pod", "container"):
                value = str(row.get(key) or "").strip()
                if value and value not in values:
                    values.append(value)
                    break
        return values

    def _build_map_default_payload(
        self,
        *,
        rows_chunk: List[Dict[str, Any]],
        batch_number: int,
        batch_period_start: Optional[str],
        batch_period_end: Optional[str],
        raw_summary_text: str = "",
    ) -> Dict[str, Any]:
        source_query = str(self.prompt_context.get("sql_query") or "").strip()
        source_services = self._derive_source_services(rows_chunk)
        total_logs = len(rows_chunk)
        timeline_seed = []
        if raw_summary_text:
            timeline_seed = [
                {
                    "id": "evt-raw-summary",
                    "timestamp": batch_period_start or _EARLIEST_PERIOD_START,
                    "source": source_services[0] if source_services else "unknown",
                    "description": "LLM вернула неструктурированный MAP summary; сохранён raw-текст.",
                    "severity": "low",
                    "importance": 0.2,
                    "evidence_type": "HYPOTHESIS",
                    "evidence_quote": "",
                    "tags": ["raw_summary", "degraded"],
                }
            ]
        noise_ratio = 1.0 if total_logs <= 0 else max(0.0, min(1.0, 1.0 - (len(timeline_seed) / float(total_logs))))
        return {
            "context": {
                "batch_id": f"batch-{int(batch_number):06d}",
                "time_range_start": batch_period_start or _EARLIEST_PERIOD_START,
                "time_range_end": batch_period_end or _EARLIEST_PERIOD_START,
                "total_log_entries": total_logs,
                "source_query": [source_query] if source_query else [],
                "source_services": source_services,
            },
            "timeline": timeline_seed,
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {
                "affected_services": [],
                "affected_operations": [],
                "error_counts": [],
                "degradation_period_start": "",
                "degradation_period_end": "",
            },
            "conflicts": [],
            "data_quality": {
                "is_empty": len(timeline_seed) == 0,
                "noise_ratio": noise_ratio,
                "has_gaps": False,
                "gap_periods": [],
                "notes": (
                    f"Raw LLM map summary (для справки): {raw_summary_text}"
                    if raw_summary_text
                    else ""
                ),
            },
            "preliminary_recommendations": [],
        }

    def _normalize_map_summary_payload(
        self,
        *,
        raw_summary_text: str,
        rows_chunk: List[Dict[str, Any]],
        batch_number: int,
        batch_period_start: Optional[str],
        batch_period_end: Optional[str],
    ) -> Tuple[MapBatchSummaryModel, Optional[str]]:
        payload = self._extract_json_payload(raw_summary_text)
        return self._normalize_map_summary_payload_from_dict(
            payload=payload,
            raw_summary_text=raw_summary_text,
            rows_chunk=rows_chunk,
            batch_number=batch_number,
            batch_period_start=batch_period_start,
            batch_period_end=batch_period_end,
        )

    def _normalize_map_summary_payload_from_dict(
        self,
        *,
        payload: Optional[Dict[str, Any]],
        raw_summary_text: str,
        rows_chunk: List[Dict[str, Any]],
        batch_number: int,
        batch_period_start: Optional[str],
        batch_period_end: Optional[str],
    ) -> Tuple[MapBatchSummaryModel, Optional[str]]:
        parse_error: Optional[str] = None
        if payload is None:
            parse_error = "LLM map summary is not valid JSON."
            payload = {}
        defaults = self._build_map_default_payload(
            rows_chunk=rows_chunk,
            batch_number=batch_number,
            batch_period_start=batch_period_start,
            batch_period_end=batch_period_end,
            raw_summary_text=raw_summary_text if parse_error else "",
        )
        merged: Dict[str, Any] = dict(payload)
        merged_context = dict(defaults["context"])
        merged_context.update(payload.get("context") or {})
        # Programmatic context is authoritative.
        merged_context["batch_id"] = defaults["context"]["batch_id"]
        merged_context["time_range_start"] = defaults["context"]["time_range_start"]
        merged_context["time_range_end"] = defaults["context"]["time_range_end"]
        merged_context["total_log_entries"] = defaults["context"]["total_log_entries"]
        merged_context["source_query"] = defaults["context"]["source_query"]
        merged_context["source_services"] = defaults["context"]["source_services"]
        merged["context"] = merged_context
        for key in (
            "timeline",
            "causal_links",
            "alert_refs",
            "hypotheses",
            "pinned_facts",
            "gaps",
            "conflicts",
            "preliminary_recommendations",
        ):
            if key not in merged or not isinstance(merged.get(key), list):
                merged[key] = defaults[key]
        if "impact" not in merged or not isinstance(merged.get("impact"), dict):
            merged["impact"] = defaults["impact"]
        if "data_quality" not in merged or not isinstance(merged.get("data_quality"), dict):
            merged["data_quality"] = defaults["data_quality"]
        try:
            model = MapBatchSummaryModel.model_validate(merged)
        except ValidationError as exc:
            parse_error = str(exc)
            degraded = self._build_map_default_payload(
                rows_chunk=rows_chunk,
                batch_number=batch_number,
                batch_period_start=batch_period_start,
                batch_period_end=batch_period_end,
                raw_summary_text=raw_summary_text,
            )
            degraded["data_quality"]["notes"] = (
                f"Validation error: {parse_error}\n\n"
                f"{degraded['data_quality'].get('notes', '')}".strip()
            )
            model = MapBatchSummaryModel.model_validate(degraded)
        return model, parse_error

    @staticmethod
    def _render_map_summary_json(model: MapBatchSummaryModel) -> str:
        return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)

    def _map_schema_retry_prompt(self, *, raw_summary_text: str, error_text: str) -> str:
        schema_hint = (
            "{"
            "\"context\": {...}, "
            "\"timeline\": [...], "
            "\"causal_links\": [...], "
            "\"alert_refs\": [...], "
            "\"hypotheses\": [...], "
            "\"pinned_facts\": [...], "
            "\"gaps\": [...], "
            "\"impact\": {...}, "
            "\"conflicts\": [...], "
            "\"data_quality\": {...}, "
            "\"preliminary_recommendations\": [...]"
            "}"
        )
        return "\n".join(
            [
                "Исправь ответ в СТРОГИЙ JSON без markdown и без пояснений.",
                "Нужен ровно один JSON-объект по схеме:",
                schema_hint,
                "",
                "Ошибка валидации предыдущего ответа:",
                str(error_text or "unknown validation error"),
                "",
                "Предыдущий ответ:",
                str(raw_summary_text or ""),
            ]
        )

    def _normalize_map_summary_response(
        self,
        *,
        raw_summary_text: str,
        rows_chunk: List[Dict[str, Any]],
        batch_number: int,
        batch_period_start: Optional[str],
        batch_period_end: Optional[str],
        allow_schema_retry: bool = True,
    ) -> Tuple[str, Dict[str, Any], int]:
        model, parse_error = self._normalize_map_summary_payload(
            raw_summary_text=raw_summary_text,
            rows_chunk=rows_chunk,
            batch_number=batch_number,
            batch_period_start=batch_period_start,
            batch_period_end=batch_period_end,
        )
        if parse_error:
            logger.warning(
                "MAP batch %s: initial schema parse failed: %s | raw_preview=%s",
                batch_number,
                parse_error,
                _preview_text(raw_summary_text, max_chars=900),
            )
        llm_extra_calls = 0
        max_schema_retries = 2
        if parse_error and allow_schema_retry:
            current_text = str(raw_summary_text or "")
            for attempt in range(1, max_schema_retries + 1):
                self._emit_progress(
                    "map_schema_retry",
                    {
                        "batch_number": int(batch_number),
                        "attempt": attempt,
                        "reason": parse_error,
                        "batch_period_start": batch_period_start,
                        "batch_period_end": batch_period_end,
                        "raw_preview": _preview_text(current_text, max_chars=900),
                    },
                )
                retry_prompt = self._map_schema_retry_prompt(
                    raw_summary_text=current_text,
                    error_text=parse_error,
                )
                current_text = str(self.llm_call(retry_prompt) or "")
                llm_extra_calls += 1
                logger.info(
                    "MAP batch %s: schema-repair attempt %s/%s finished; reply_len=%s",
                    batch_number,
                    attempt,
                    max_schema_retries,
                    len(current_text),
                )
                model, parse_error = self._normalize_map_summary_payload(
                    raw_summary_text=current_text,
                    rows_chunk=rows_chunk,
                    batch_number=batch_number,
                    batch_period_start=batch_period_start,
                    batch_period_end=batch_period_end,
                )
                if not parse_error:
                    self._emit_progress(
                        "map_schema_recovered",
                        {
                            "batch_number": int(batch_number),
                            "attempt": attempt,
                            "batch_period_start": batch_period_start,
                            "batch_period_end": batch_period_end,
                        },
                    )
                    logger.info(
                        "MAP batch %s: schema recovered on attempt %s/%s",
                        batch_number,
                        attempt,
                        max_schema_retries,
                    )
                    break
        if parse_error:
            self._emit_progress(
                "map_schema_degraded",
                {
                    "batch_number": int(batch_number),
                    "reason": parse_error,
                    "batch_period_start": batch_period_start,
                    "batch_period_end": batch_period_end,
                    "raw_preview": _preview_text(raw_summary_text, max_chars=1200),
                    "raw_len": len(str(raw_summary_text or "")),
                },
            )
            logger.warning(
                "MAP batch %s: schema degraded after retries; reason=%s | raw_preview=%s",
                batch_number,
                parse_error,
                _preview_text(raw_summary_text, max_chars=900),
            )
        json_text = self._truncate(
            self._render_map_summary_json(model),
            self.config.max_summary_chars,
        )
        return json_text, model.model_dump(mode="json"), llm_extra_calls

    def _normalize_map_model_from_instructor(
        self,
        *,
        model: MapBatchSummaryModel,
        rows_chunk: List[Dict[str, Any]],
        batch_number: int,
        batch_period_start: Optional[str],
        batch_period_end: Optional[str],
    ) -> Tuple[str, Dict[str, Any]]:
        normalized, parse_error = self._normalize_map_summary_payload_from_dict(
            payload=model.model_dump(mode="json"),
            raw_summary_text="",
            rows_chunk=rows_chunk,
            batch_number=batch_number,
            batch_period_start=batch_period_start,
            batch_period_end=batch_period_end,
        )
        if parse_error:
            logger.warning(
                "MAP batch %s: instructor payload normalization warning: %s",
                batch_number,
                parse_error,
            )
        json_text = self._truncate(
            self._render_map_summary_json(normalized),
            self.config.max_summary_chars,
        )
        return json_text, normalized.model_dump(mode="json")

    def _emit_progress(self, event: str, payload: Dict[str, Any]) -> None:
        logger.info(
            "summarizer.progress event=%s payload=%s",
            event,
            _payload_for_progress_log(payload),
        )
        if self.on_progress is None:
            return
        self.on_progress(event, payload)

    @staticmethod
    def _is_400_bad_request_fallback(summary_text: str) -> bool:
        raw = str(summary_text or "").strip().lower()
        if not _is_llm_error_stub(raw):
            return False
        return (
            ("400 client error" in raw and "bad request" in raw)
            or "code: 400" in raw
            or "status code 400" in raw
        )

    def _summarize_rows_with_auto_shrink(
        self,
        *,
        rows_chunk: List[Dict[str, Any]],
        columns: Sequence[str],
        period_start: str,
        period_end: str,
        batch_number: int,
        total_batches: Optional[int],
        depth: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        batch_period_start, batch_period_end = _extract_batch_period(rows_chunk)
        prompt = self._build_chunk_prompt(
            period_start=period_start,
            period_end=period_end,
            columns=columns,
            rows=rows_chunk,
            batch_number=batch_number,
            total_batches=total_batches,
        )
        min_chunk = max(int(getattr(self.config, "min_llm_chunk_rows", 20) or 20), 1)
        max_shrink_rounds = max(int(getattr(self.config, "max_shrink_rounds", 6) or 6), 0)
        auto_shrink = bool(getattr(self.config, "auto_shrink_on_400", True))
        instructor_path_used = False
        try:
            if self._instructor_enabled() and not self._instructor_map_disabled_due_grammar:
                instructor_path_used = True
                structured_model, instructor_attempts = self._call_structured_with_instructor(
                    prompt=prompt,
                    response_model=MapBatchSummaryModel,
                    stage="map",
                )
                chunk_summary, chunk_summary_structured = self._normalize_map_model_from_instructor(
                    model=structured_model,
                    rows_chunk=rows_chunk,
                    batch_number=batch_number,
                    batch_period_start=batch_period_start,
                    batch_period_end=batch_period_end,
                )
                return (
                    [
                        {
                            "summary": chunk_summary,
                            "summary_structured": chunk_summary_structured,
                            "rows": list(rows_chunk),
                            "batch_period_start": batch_period_start,
                            "batch_period_end": batch_period_end,
                        }
                    ],
                    max(instructor_attempts, 1),
                )
            llm_text = self.llm_call(prompt)
        except Exception as exc:
            logger.warning(
                "MAP batch %s: LLM call failed | rows=%s | depth=%s | error=%s",
                batch_number,
                len(rows_chunk),
                depth,
                exc,
            )
            if instructor_path_used and _is_invalid_grammar_request_exception(exc):
                logger.warning(
                    "MAP batch %s: Instructor grammar is unsupported by gateway for MAP schema. "
                    "Switching MAP to legacy mode for this run.",
                    batch_number,
                )
                self._instructor_map_disabled_due_grammar = True
                self._emit_progress(
                    "map_instructor_fallback",
                    {
                        "batch_number": batch_number,
                        "depth": depth,
                        "reason": "invalid_grammar_request",
                        "batch_period_start": batch_period_start,
                        "batch_period_end": batch_period_end,
                        "error": str(exc),
                    },
                )
                try:
                    llm_text = self.llm_call(prompt)
                    raw_llm_text = (llm_text or "").strip()
                    llm_calls = 2  # failed instructor attempt + legacy llm call
                    chunk_summary, chunk_summary_structured, schema_retry_calls = self._normalize_map_summary_response(
                        raw_summary_text=raw_llm_text,
                        rows_chunk=rows_chunk,
                        batch_number=batch_number,
                        batch_period_start=batch_period_start,
                        batch_period_end=batch_period_end,
                        allow_schema_retry=True,
                    )
                    llm_calls += schema_retry_calls
                    return (
                        [
                            {
                                "summary": chunk_summary,
                                "summary_structured": chunk_summary_structured,
                                "rows": list(rows_chunk),
                                "batch_period_start": batch_period_start,
                                "batch_period_end": batch_period_end,
                            }
                        ],
                        llm_calls,
                    )
                except Exception as legacy_exc:  # noqa: BLE001
                    logger.warning(
                        "MAP batch %s: legacy fallback after instructor grammar failure also failed: %s",
                        batch_number,
                        legacy_exc,
                    )
                    exc = legacy_exc
            can_shrink_on_exc = (
                auto_shrink
                and _is_400_bad_request_exception(exc)
                and len(rows_chunk) > min_chunk
                and depth < max_shrink_rounds
            )
            if not can_shrink_on_exc:
                raise
            next_size = max(min_chunk, len(rows_chunk) // 2)
            if next_size >= len(rows_chunk):
                raise
            self._emit_progress(
                "map_batch_resize",
                {
                    "batch_number": batch_number,
                    "depth": depth,
                    "old_chunk_size": len(rows_chunk),
                    "new_chunk_size": next_size,
                    "reason": "llm_400_bad_request_exception",
                    "batch_period_start": batch_period_start,
                    "batch_period_end": batch_period_end,
                    "error": str(exc),
                },
            )
            logger.info(
                "MAP batch %s: auto-shrink on exception | depth=%s | %s -> %s",
                batch_number,
                depth,
                len(rows_chunk),
                next_size,
            )
            merged_items: List[Dict[str, Any]] = []
            llm_calls = 1
            for start in range(0, len(rows_chunk), next_size):
                sub_rows = rows_chunk[start : start + next_size]
                sub_items, sub_calls = self._summarize_rows_with_auto_shrink(
                    rows_chunk=sub_rows,
                    columns=columns,
                    period_start=period_start,
                    period_end=period_end,
                    batch_number=batch_number,
                    total_batches=total_batches,
                    depth=depth + 1,
                )
                merged_items.extend(sub_items)
                llm_calls += sub_calls
            return merged_items, llm_calls

        if instructor_path_used:
            # Defensive: instructor path returns above; keep guard explicit.
            raise RuntimeError("Instructor MAP path reached unexpected branch.")

        raw_llm_text = (llm_text or "").strip()
        llm_calls = 1
        chunk_summary, chunk_summary_structured, schema_retry_calls = self._normalize_map_summary_response(
            raw_summary_text=raw_llm_text,
            rows_chunk=rows_chunk,
            batch_number=batch_number,
            batch_period_start=batch_period_start,
            batch_period_end=batch_period_end,
            allow_schema_retry=True,
        )
        llm_calls += schema_retry_calls

        can_shrink = (
            auto_shrink
            and self._is_400_bad_request_fallback(raw_llm_text)
            and len(rows_chunk) > min_chunk
            and depth < max_shrink_rounds
        )
        if not can_shrink:
            return (
                [
                    {
                        "summary": chunk_summary,
                        "summary_structured": chunk_summary_structured,
                        "rows": list(rows_chunk),
                        "batch_period_start": batch_period_start,
                        "batch_period_end": batch_period_end,
                    }
                ],
                llm_calls,
            )

        next_size = max(min_chunk, len(rows_chunk) // 2)
        if next_size >= len(rows_chunk):
            return (
                [
                    {
                        "summary": chunk_summary,
                        "summary_structured": chunk_summary_structured,
                        "rows": list(rows_chunk),
                        "batch_period_start": batch_period_start,
                        "batch_period_end": batch_period_end,
                    }
                ],
                llm_calls,
            )

        self._emit_progress(
            "map_batch_resize",
            {
                "batch_number": batch_number,
                "depth": depth,
                "old_chunk_size": len(rows_chunk),
                "new_chunk_size": next_size,
                "reason": "llm_400_bad_request",
                "batch_period_start": batch_period_start,
                "batch_period_end": batch_period_end,
            },
        )
        logger.info(
            "MAP batch %s: auto-shrink on 400 fallback text | depth=%s | %s -> %s",
            batch_number,
            depth,
            len(rows_chunk),
            next_size,
        )

        merged_items: List[Dict[str, Any]] = []
        for sub_start in range(0, len(rows_chunk), next_size):
            sub_rows = rows_chunk[sub_start : sub_start + next_size]
            sub_items, sub_calls = self._summarize_rows_with_auto_shrink(
                rows_chunk=sub_rows,
                columns=columns,
                period_start=period_start,
                period_end=period_end,
                batch_number=batch_number,
                total_batches=total_batches,
                depth=depth + 1,
            )
            merged_items.extend(sub_items)
            llm_calls += sub_calls
        return merged_items, llm_calls

    def summarize_period(
        self,
        *,
        period_start: str,
        period_end: str,
        columns: Sequence[str],
        total_rows_estimate: Optional[int] = None,
    ) -> SummarizationResult:
        self._validate_iso_datetime(period_start)
        self._validate_iso_datetime(period_end)
        if not columns:
            raise ValueError("columns must not be empty")

        offset = 0
        pages_fetched = 0
        rows_processed = 0
        llm_calls = 0
        map_summaries: List[str] = []
        map_batches: List[Dict[str, Any]] = []
        map_batch_index = 0
        rows_mapped = 0
        estimated_batch_total: Optional[int] = None
        if total_rows_estimate is not None and total_rows_estimate > 0:
            estimated_batch_total = max(
                int(math.ceil(float(total_rows_estimate) / float(max(self.config.llm_chunk_rows, 1)))),
                1,
            )
        if bool(getattr(self.config, "auto_shrink_on_400", True)):
            # Real batch count can grow dynamically after split-on-400.
            estimated_batch_total = None
        if bool(getattr(self.config, "use_new_algorithm", False)):
            # Token-based map splitting can further split chunks dynamically.
            estimated_batch_total = None
        seen_sources: List[str] = []  # ordered unique sources seen across all pages
        self._emit_progress(
            "map_start",
            {
                "rows_processed": 0,
                "rows_total": total_rows_estimate,
                "use_new_algorithm": bool(getattr(self.config, "use_new_algorithm", False)),
                "use_instructor": bool(getattr(self.config, "use_instructor", True)),
                "instructor_runtime_available": bool(instructor is not None),
                "model_supports_tool_calling": bool(
                    getattr(self.config, "model_supports_tool_calling", True)
                ),
            },
        )
        logger.info(
            "MAP start | period=[%s, %s) | page_limit=%s | llm_chunk_rows=%s | min_llm_chunk_rows=%s | auto_shrink_on_400=%s | max_shrink_rounds=%s | map_workers=%s | use_new_algorithm=%s",
            period_start,
            period_end,
            self.config.page_limit,
            self.config.llm_chunk_rows,
            int(getattr(self.config, "min_llm_chunk_rows", 20) or 20),
            bool(getattr(self.config, "auto_shrink_on_400", True)),
            int(getattr(self.config, "max_shrink_rounds", 6) or 6),
            int(getattr(self.config, "map_workers", 1) or 1),
            bool(getattr(self.config, "use_new_algorithm", False)),
        )

        # Each pending entry: (Future | None, batch_index, rows_count, bp_start, bp_end, rows_snapshot)
        # rows_snapshot is kept only for the progress callback; raw rows are not stored otherwise.
        _PendingItem = Tuple[
            Optional[Future],  # type: ignore[type-arg]
            int,               # batch_index
            int,               # rows_count
            Optional[str],     # batch_period_start
            Optional[str],     # batch_period_end
            List[Dict[str, Any]],  # rows snapshot for progress callback
        ]
        pending_items: List[_PendingItem] = []
        parallel = self.config.map_workers > 1 and not bool(
            getattr(self.config, "auto_shrink_on_400", True)
        ) and not self._instructor_enabled()
        if self.config.map_workers > 1 and not parallel:
            self._emit_progress(
                "map_parallel_disabled",
                {
                    "reason": (
                        "auto_shrink_on_400"
                        if bool(getattr(self.config, "auto_shrink_on_400", True))
                        else "instructor_structured_mode"
                    ),
                    "map_workers_requested": int(self.config.map_workers),
                },
            )
        executor = ThreadPoolExecutor(max_workers=self.config.map_workers) if parallel else None

        try:
            while True:
                page = self.db_fetch_page(
                    columns=columns,
                    period_start=period_start,
                    period_end=period_end,
                    limit=self.config.page_limit,
                    offset=offset,
                )
                if not page:
                    break

                pages_fetched += 1
                rows_processed += len(page)
                offset += len(page)
                for _row in page:
                    _src = str(_row.get("_source") or "")
                    if _src and _src not in seen_sources:
                        seen_sources.append(_src)
                self._emit_progress(
                    "page_fetched",
                    {
                        "page_index": pages_fetched,
                        "page_rows": len(page),
                        "rows_fetched": rows_processed,
                        "rows_total": total_rows_estimate,
                    },
                )
                logger.info(
                    "Page fetched | page_index=%s | page_rows=%s | rows_fetched=%s | next_offset=%s",
                    pages_fetched,
                    len(page),
                    rows_processed,
                    offset,
                )

                if bool(getattr(self.config, "use_new_algorithm", False)):
                    # Updated algorithm: one DB page == one map batch (no local token pre-splitting).
                    base_chunks: List[List[Dict[str, Any]]] = [list(page)]
                else:
                    base_chunks = [
                        page[i : i + self.config.llm_chunk_rows]
                        for i in range(0, len(page), self.config.llm_chunk_rows)
                    ]

                for base_chunk in base_chunks:
                    rows_sub_chunks = self._split_rows_for_map(
                        rows=list(base_chunk),
                        columns=columns,
                    )
                    for rows_chunk in rows_sub_chunks:
                        next_batch_index = map_batch_index
                        batch_period_start, batch_period_end = _extract_batch_period(rows_chunk)
                        self._emit_progress(
                            "map_batch_start",
                            {
                                "batch_index": next_batch_index,
                                "batch_total": estimated_batch_total,
                                "batch_logs_count": len(rows_chunk),
                                "batch_period_start": batch_period_start,
                                "batch_period_end": batch_period_end,
                                "rows_processed": rows_mapped,
                                "rows_total": total_rows_estimate,
                            },
                        )
                        logger.info(
                            "MAP batch start | batch=%s/%s | rows=%s | period=%s..%s",
                            next_batch_index + 1,
                            estimated_batch_total if estimated_batch_total is not None else "dynamic",
                            len(rows_chunk),
                            batch_period_start,
                            batch_period_end,
                        )
                        prompt = self._build_chunk_prompt(
                            period_start=period_start,
                            period_end=period_end,
                            columns=columns,
                            rows=rows_chunk,
                            batch_number=next_batch_index + 1,
                            total_batches=estimated_batch_total,
                        )
                        future: Optional[Future] = None  # type: ignore[type-arg]
                        if executor is not None:
                            future = executor.submit(self.llm_call, prompt)
                        else:
                            # Sequential: run inline with optional auto-shrink on 400.
                            summarized_items, map_llm_calls = self._summarize_rows_with_auto_shrink(
                                rows_chunk=list(rows_chunk),
                                columns=columns,
                                period_start=period_start,
                                period_end=period_end,
                                batch_number=next_batch_index + 1,
                                total_batches=estimated_batch_total,
                            )
                            llm_calls += map_llm_calls
                            for item in summarized_items:
                                item_rows = list(item.get("rows") or [])
                                item_summary = str(item.get("summary") or "").strip()
                                item_summary_structured = dict(item.get("summary_structured") or {})
                                item_bp_start = item.get("batch_period_start")
                                item_bp_end = item.get("batch_period_end")
                                current_batch_index = map_batch_index
                                map_summaries.append(item_summary)
                                if self.config.keep_map_batches_in_memory:
                                    map_batches.append(
                                        {
                                            "batch_index": current_batch_index,
                                            "rows_count": len(item_rows),
                                            "summary": item_summary,
                                            "summary_structured": item_summary_structured,
                                            "batch_period_start": item_bp_start,
                                            "batch_period_end": item_bp_end,
                                        }
                                    )
                                rows_mapped += len(item_rows)
                                self._emit_progress(
                                    "map_batch",
                                    {
                                        "batch_index": current_batch_index,
                                        "batch_total": estimated_batch_total,
                                        "batch_summary": item_summary,
                                        "batch_summary_structured": item_summary_structured,
                                        "batch_logs_count": len(item_rows),
                                        "batch_logs": item_rows,
                                        "batch_period_start": item_bp_start,
                                        "batch_period_end": item_bp_end,
                                        "rows_processed": rows_mapped,
                                        "rows_total": total_rows_estimate,
                                    },
                                )
                                logger.info(
                                    "MAP batch done | batch=%s/%s | rows=%s | llm_calls_total=%s | timeline=%s | hypotheses=%s | alerts=%s",
                                    current_batch_index + 1,
                                    estimated_batch_total if estimated_batch_total is not None else "dynamic",
                                    len(item_rows),
                                    llm_calls,
                                    len(item_summary_structured.get("timeline") or []),
                                    len(item_summary_structured.get("hypotheses") or []),
                                    len(item_summary_structured.get("alert_refs") or []),
                                )
                                map_batch_index += 1
                            continue  # skip appending to pending_items again

                        # Parallel mode: store future for later collection
                        pending_items.append((
                            future,
                            next_batch_index,
                            len(rows_chunk),
                            batch_period_start,
                            batch_period_end,
                            list(rows_chunk),
                        ))
                        map_batch_index += 1

                if len(page) < self.config.page_limit:
                    break

        except Exception:
            if executor is not None:
                executor.shutdown(wait=False)
            raise
        else:
            if executor is not None:
                executor.shutdown(wait=True)

        # --- Parallel mode: collect futures in submission order ---
        if parallel and pending_items:
            for fut, batch_idx, nrows, bp_start, bp_end, rows_snap in pending_items:
                if fut is None:
                    continue  # sequential items already handled above
                try:
                    result_text = fut.result()
                except Exception as exc:
                    logger.exception("Parallel LLM call failed on batch %s", batch_idx)
                    raise
                chunk_summary, chunk_summary_structured, extra_calls = self._normalize_map_summary_response(
                    raw_summary_text=(result_text or "").strip(),
                    rows_chunk=rows_snap,
                    batch_number=batch_idx + 1,
                    batch_period_start=bp_start,
                    batch_period_end=bp_end,
                    allow_schema_retry=False,
                )
                map_summaries.append(chunk_summary)
                if self.config.keep_map_batches_in_memory:
                    map_batches.append({
                        "batch_index": batch_idx,
                        "rows_count": nrows,
                        "summary": chunk_summary,
                        "summary_structured": chunk_summary_structured,
                        "batch_period_start": bp_start,
                        "batch_period_end": bp_end,
                    })
                rows_mapped += nrows
                llm_calls += 1 + extra_calls
                self._emit_progress(
                    "map_batch",
                    {
                        "batch_index": batch_idx,
                        "batch_total": estimated_batch_total,
                        "batch_summary": chunk_summary,
                        "batch_summary_structured": chunk_summary_structured,
                        "batch_logs_count": nrows,
                        "batch_logs": rows_snap,
                        "batch_period_start": bp_start,
                        "batch_period_end": bp_end,
                        "rows_processed": rows_mapped,
                        "rows_total": total_rows_estimate,
                    },
                )
                logger.info(
                    "MAP batch done (parallel) | batch=%s/%s | rows=%s | llm_calls_total=%s | timeline=%s | hypotheses=%s | alerts=%s",
                    batch_idx + 1,
                    estimated_batch_total if estimated_batch_total is not None else "dynamic",
                    nrows,
                    llm_calls,
                    len(chunk_summary_structured.get("timeline") or []),
                    len(chunk_summary_structured.get("hypotheses") or []),
                    len(chunk_summary_structured.get("alert_refs") or []),
                )

        if not map_summaries:
            self._emit_progress(
                "map_done",
                {
                    "batch_total": 0,
                    "rows_processed": 0,
                    "rows_total": total_rows_estimate,
                },
            )
            return SummarizationResult(
                summary="Нет логов за указанный период.",
                pages_fetched=pages_fetched,
                rows_processed=rows_processed,
                llm_calls=llm_calls,
                chunk_summaries=0,
                reduce_rounds=0,
                map_summaries=[],
                map_batches=[],
            )

        self._emit_progress(
            "map_done",
            {
                "batch_total": map_batch_index,
                "rows_processed": rows_mapped,
                "rows_total": total_rows_estimate,
            },
        )
        logger.info(
            "MAP done | batches=%s | rows_mapped=%s | llm_calls=%s",
            map_batch_index,
            rows_mapped,
            llm_calls,
        )
        self._emit_progress(
            "reduce_start",
            {
                "batch_total": map_batch_index,
                "rows_processed": rows_mapped,
                "rows_total": total_rows_estimate,
            },
        )
        final_summary, reduce_calls, reduce_rounds = self._reduce_summaries(
            chunk_summaries=map_summaries,
            period_start=period_start,
            period_end=period_end,
            sources=seen_sources or None,
        )
        self._emit_progress(
            "reduce_done",
            {
                "summary": final_summary,
                "rows_processed": rows_mapped,
                "rows_total": total_rows_estimate,
            },
        )
        llm_calls += reduce_calls
        logger.info(
            "REDUCE done | reduce_calls=%s | reduce_rounds=%s | llm_calls_total=%s",
            reduce_calls,
            reduce_rounds,
            llm_calls,
        )

        # Freeform narrative: one more LLM pass to produce a human-readable story for SRE team
        freeform_summary = ""
        if final_summary and final_summary != "Нет логов за указанный период.":
            self._emit_progress("freeform_start", {"rows_processed": rows_mapped})
            try:
                freeform_prompt = self._build_freeform_prompt(
                    period_start=period_start,
                    period_end=period_end,
                    structured_summary=final_summary,
                    map_summaries=map_summaries,
                )
                freeform_summary = self.llm_call(freeform_prompt).strip()
                llm_calls += 1
            except Exception:
                logger.warning("Freeform narrative generation failed, skipping")
            self._emit_progress("freeform_done", {"freeform_summary": freeform_summary})

        result_map_summaries = map_summaries if self.config.keep_map_summaries_in_result else []
        return SummarizationResult(
            summary=final_summary,
            freeform_summary=freeform_summary,
            pages_fetched=pages_fetched,
            rows_processed=rows_processed,
            llm_calls=llm_calls,
            chunk_summaries=len(map_summaries),
            reduce_rounds=reduce_rounds,
            map_summaries=result_map_summaries,
            map_batches=map_batches,
        )

    def _reduce_structured_group(
        self,
        *,
        summaries: List[IncidentSummary],
        period_start: str,
        period_end: str,
        reduce_round: int,
        group_index: int,
        group_total: int,
        sources: Optional[List[str]] = None,
        depth: int = 0,
    ) -> tuple[IncidentSummary, int]:
        if len(summaries) == 1:
            return summaries[0], 0
        if depth > 12:
            # Safety net: deterministic fallback to guarantee convergence.
            merged_alerts = self._deterministic_merge_alert_refs(summaries)
            total_logs = sum(max(int(s.context.total_log_entries), 0) for s in summaries)
            fallback = IncidentSummary(
                context=IncidentContext(
                    batch_id=f"reduce-fallback-r{reduce_round}-g{group_index}",
                    time_range_start=period_start,
                    time_range_end=period_end,
                    total_log_entries=total_logs,
                    source_query=[],
                    source_services=[],
                ),
                timeline=[],
                causal_links=[],
                alert_refs=merged_alerts,
                hypotheses=[],
                pinned_facts=[],
                gaps=[],
                impact=IncidentImpact(),
                conflicts=[],
                data_quality=IncidentDataQuality(
                    is_empty=True,
                    noise_ratio=1.0,
                    notes="Deterministic fallback merge (max recursion depth reached).",
                ),
                preliminary_recommendations=[],
            )
            return fallback, 0

        serialized_group = [self._incident_summary_to_json(item) for item in summaries]
        merged_alerts = self._deterministic_merge_alert_refs(summaries)
        prompt = self._build_reduce_prompt_with_premerged_alerts(
            period_start=period_start,
            period_end=period_end,
            reduce_round=reduce_round,
            summaries=serialized_group,
            premerged_alert_refs=merged_alerts,
            sources=sources,
        )

        calls = 0
        try:
            if self._instructor_enabled():
                parsed, calls = self._call_structured_with_instructor(
                    prompt=prompt,
                    response_model=IncidentSummary,
                    stage="reduce",
                )
                return parsed, calls

            raw = str(self.llm_call(prompt) or "").strip()
            calls += 1
            parsed = self._parse_incident_summary_text(
                raw,
                fallback_batch_id=f"reduce-r{reduce_round}-g{group_index}",
                fallback_start=period_start,
                fallback_end=period_end,
            )
            if parsed is not None:
                return parsed, calls

            # Schema-repair retry for legacy non-instructor path.
            retry_prompt = "\n".join(
                [
                    "Return STRICT JSON only (no markdown, no explanations).",
                    "The JSON must validate against IncidentSummary schema.",
                    "Fix invalid references and required fields.",
                    "",
                    "Previous invalid response:",
                    raw,
                ]
            )
            raw_retry = str(self.llm_call(retry_prompt) or "").strip()
            calls += 1
            parsed_retry = self._parse_incident_summary_text(
                raw_retry,
                fallback_batch_id=f"reduce-r{reduce_round}-g{group_index}",
                fallback_start=period_start,
                fallback_end=period_end,
            )
            if parsed_retry is not None:
                return parsed_retry, calls

            # If response is plain text, keep compatibility by giving control back to legacy reducer.
            raise ValueError("structured_reduce_invalid_json")
        except Exception as exc:  # noqa: BLE001
            if (
                self._is_context_overflow_error(exc)
                or _is_400_bad_request_exception(exc)
                or _is_read_timeout_exception(exc)
            ) and len(summaries) > 1:
                mid = max(len(summaries) // 2, 1)
                left, left_calls = self._reduce_structured_group(
                    summaries=summaries[:mid],
                    period_start=period_start,
                    period_end=period_end,
                    reduce_round=reduce_round,
                    group_index=group_index,
                    group_total=group_total,
                    sources=sources,
                    depth=depth + 1,
                )
                right, right_calls = self._reduce_structured_group(
                    summaries=summaries[mid:],
                    period_start=period_start,
                    period_end=period_end,
                    reduce_round=reduce_round,
                    group_index=group_index,
                    group_total=group_total,
                    sources=sources,
                    depth=depth + 1,
                )
                compression_calls = 0
                if len(summaries) == 2:
                    # Updated algorithm: when a small group still overflows, proactively
                    # compress single summaries using COMPRESSION prompt and retry.
                    left, c_left = self._compress_summary_on_overflow(left)
                    right, c_right = self._compress_summary_on_overflow(right)
                    compression_calls += c_left + c_right
                combined, combined_calls = self._reduce_structured_group(
                    summaries=[left, right],
                    period_start=period_start,
                    period_end=period_end,
                    reduce_round=reduce_round,
                    group_index=group_index,
                    group_total=group_total,
                    sources=sources,
                    depth=depth + 1,
                )
                return combined, calls + left_calls + right_calls + compression_calls + combined_calls
            raise

    def _reduce_summaries_structured(
        self,
        *,
        chunk_summaries: List[str],
        period_start: str,
        period_end: str,
        sources: Optional[List[str]] = None,
    ) -> Optional[tuple[str, int, int]]:
        if len(chunk_summaries) <= 1:
            return None
        parsed_inputs: List[IncidentSummary] = []
        for idx, text in enumerate(chunk_summaries, start=1):
            parsed = self._parse_incident_summary_text(
                str(text or ""),
                fallback_batch_id=f"map-{idx:06d}",
                fallback_start=period_start,
                fallback_end=period_end,
            )
            if parsed is None:
                # Not a structured payload => keep legacy behavior.
                return None
            if not self._summary_is_empty(parsed):
                parsed_inputs.append(parsed)

        if not parsed_inputs:
            return (
                "В предоставленных логах не обнаружено событий, связанных с инцидентом. "
                "Возможные причины: логи не покрывают нужный временной диапазон, "
                "релевантные сервисы не представлены в выборке, уровень логирования недостаточен.",
                0,
                0,
            )

        llm_calls = 0
        reduce_rounds = 0
        current = parsed_inputs
        while len(current) > 1:
            reduce_rounds += 1
            if reduce_rounds > self.config.max_reduce_rounds:
                raise RuntimeError("Exceeded max reduce rounds")

            groups = self._group_structured_summaries_by_budget(current)

            next_level: List[IncidentSummary] = []
            for group_index, group in enumerate(groups):
                self._emit_progress(
                    "reduce_group_start",
                    {
                        "reduce_round": reduce_rounds,
                        "group_index": group_index,
                        "group_total": len(groups),
                        "group_size": len(group),
                    },
                )
                try:
                    merged, group_calls = self._reduce_structured_group(
                        summaries=group,
                        period_start=period_start,
                        period_end=period_end,
                        reduce_round=reduce_rounds,
                        group_index=group_index + 1,
                        group_total=len(groups),
                        sources=sources,
                    )
                except ValueError as exc:
                    if str(exc) == "structured_reduce_invalid_json":
                        # Fallback to legacy reducer for compatibility with plain-text LLM outputs.
                        return None
                    raise
                llm_calls += group_calls
                next_level.append(merged)
                self._emit_progress(
                    "reduce_group_done",
                    {
                        "reduce_round": reduce_rounds,
                        "group_index": group_index,
                        "group_total": len(groups),
                        "group_size": len(group),
                    },
                )
            current = next_level

        final_text = self._incident_summary_to_json(current[0])
        return final_text, llm_calls, reduce_rounds

    def _reduce_summaries(
        self,
        *,
        chunk_summaries: List[str],
        period_start: str,
        period_end: str,
        sources: Optional[List[str]] = None,
    ) -> tuple[str, int, int]:
        if bool(getattr(self.config, "use_new_algorithm", False)):
            structured_result = self._reduce_summaries_structured(
                chunk_summaries=chunk_summaries,
                period_start=period_start,
                period_end=period_end,
                sources=sources,
            )
            if structured_result is not None:
                return structured_result
        if len(chunk_summaries) == 1:
            return chunk_summaries[0], 0, 0
        if not self.config.adaptive_reduce_on_overflow:
            return self._reduce_summaries_fixed_groups(
                chunk_summaries=chunk_summaries,
                period_start=period_start,
                period_end=period_end,
                sources=sources,
            )

        # 1) Try single-pass reduce over all map batches first.
        # 2) Only if it does not fit context (or context-overflow error), switch to adaptive shrinking.
        current = list(chunk_summaries)
        llm_calls = 0
        first_round = 1
        full_prompt = self._build_reduce_prompt(
            period_start=period_start,
            period_end=period_end,
            reduce_round=first_round,
            summaries=current,
            sources=sources,
        )
        full_fits = self._prompt_fits_budget(full_prompt)
        if full_fits:
            self._emit_progress(
                "reduce_group_start",
                {
                    "reduce_round": first_round,
                    "group_index": 0,
                    "group_total": 1,
                },
            )
            try:
                merged = self.llm_call(full_prompt).strip()
                llm_calls += 1
                if not merged:
                    merged = "Пустой ответ LLM на reduce-этапе."
                self._emit_progress(
                    "reduce_group_done",
                    {
                        "reduce_round": first_round,
                        "group_index": 0,
                        "group_total": 1,
                    },
                )
                return self._truncate(merged, self.config.max_summary_chars), llm_calls, 1
            except Exception as exc:  # noqa: BLE001
                if not (
                    self._is_context_overflow_error(exc)
                    or _is_400_bad_request_exception(exc)
                ):
                    raise
                logger.warning("reduce full-merge overflow, fallback to adaptive mode: %s", exc)
                self._emit_progress(
                    "reduce_context_fallback",
                    {
                        "reduce_round": first_round,
                        "reason": str(exc),
                    },
                )
        else:
            self._emit_progress(
                "reduce_context_fallback",
                {
                    "reduce_round": first_round,
                    "reason": (
                        "full reduce prompt exceeds reduce_prompt_max_chars="
                        f"{self.config.reduce_prompt_max_chars}"
                    ),
                },
            )

        # Adaptive mode: for each round try to merge the largest possible group.
        round_idx = 0
        while len(current) > 1:
            round_idx += 1
            if round_idx > self.config.max_reduce_rounds:
                raise RuntimeError("Exceeded max reduce rounds")

            next_level: List[str] = []
            cursor = 0
            group_index = 0
            previous_len = len(current)
            while cursor < len(current):
                remaining = len(current) - cursor
                used_size = remaining
                merged_text: Optional[str] = None

                while used_size >= 1:
                    group = current[cursor : cursor + used_size]
                    if used_size == 1:
                        # Single summary cannot be reduced further; pass through.
                        merged_text = self._truncate(
                            str(group[0]),
                            self.config.max_summary_chars,
                        )
                        break

                    prompt = self._build_reduce_prompt(
                        period_start=period_start,
                        period_end=period_end,
                        reduce_round=round_idx,
                        summaries=group,
                        sources=sources,
                    )
                    if not self._prompt_fits_budget(prompt):
                        used_size -= 1
                        continue

                    self._emit_progress(
                        "reduce_group_start",
                        {
                            "reduce_round": round_idx,
                            "group_index": group_index,
                            "group_total": previous_len,
                            "group_size": used_size,
                        },
                    )
                    try:
                        merged = self.llm_call(prompt).strip()
                        llm_calls += 1
                        if not merged:
                            merged = "Пустой ответ LLM на reduce-этапе."
                        merged_text = self._truncate(merged, self.config.max_summary_chars)
                        self._emit_progress(
                            "reduce_group_done",
                            {
                                "reduce_round": round_idx,
                                "group_index": group_index,
                                "group_total": previous_len,
                                "group_size": used_size,
                            },
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        if (
                            self._is_context_overflow_error(exc)
                            or _is_400_bad_request_exception(exc)
                        ) and used_size > 1:
                            used_size -= 1
                            continue
                        raise

                if merged_text is None:
                    # Safety net: force progress in pathological edge-cases.
                    used_size = min(2, remaining)
                    forced = "\n\n".join(str(x) for x in current[cursor : cursor + used_size])
                    merged_text = self._truncate(forced, self.config.max_summary_chars)

                next_level.append(merged_text)
                cursor += max(used_size, 1)
                group_index += 1

            if len(next_level) >= previous_len:
                # Guarantee convergence to a single summary.
                compressed: List[str] = []
                for i in range(0, len(next_level), 2):
                    pair = next_level[i : i + 2]
                    compressed.append(
                        self._truncate("\n\n".join(pair), self.config.max_summary_chars)
                    )
                next_level = compressed
            current = next_level

        return current[0], llm_calls, round_idx

    def _reduce_summaries_fixed_groups(
        self,
        *,
        chunk_summaries: List[str],
        period_start: str,
        period_end: str,
        sources: Optional[List[str]] = None,
    ) -> tuple[str, int, int]:
        round_idx = 0
        current = chunk_summaries
        llm_calls = 0
        while len(current) > 1:
            round_idx += 1
            if round_idx > self.config.max_reduce_rounds:
                raise RuntimeError("Exceeded max reduce rounds")
            next_level: List[str] = []
            groups_total = int(math.ceil(len(current) / max(self.config.reduce_group_size, 1)))
            for i in range(0, len(current), self.config.reduce_group_size):
                group = current[i : i + self.config.reduce_group_size]
                group_index = int(i / max(self.config.reduce_group_size, 1))
                self._emit_progress(
                    "reduce_group_start",
                    {
                        "reduce_round": round_idx,
                        "group_index": group_index,
                        "group_total": groups_total,
                    },
                )
                prompt = self._build_reduce_prompt(
                    period_start=period_start,
                    period_end=period_end,
                    reduce_round=round_idx,
                    summaries=group,
                    sources=sources,
                )
                merged = self.llm_call(prompt).strip()
                if not merged:
                    merged = "Пустой ответ LLM на reduce-этапе."
                next_level.append(self._truncate(merged, self.config.max_summary_chars))
                self._emit_progress(
                    "reduce_group_done",
                    {
                        "reduce_round": round_idx,
                        "group_index": group_index,
                        "group_total": groups_total,
                    },
                )
                llm_calls += 1
            current = next_level
        return current[0], llm_calls, round_idx

    def _prompt_fits_budget(self, prompt: str) -> bool:
        max_chars = max(int(getattr(self.config, "reduce_prompt_max_chars", 0)), 0)
        if max_chars <= 0:
            return True
        return len(prompt) <= max_chars

    @staticmethod
    def _is_context_overflow_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "maximum context length",
            "context length",
            "too many tokens",
            "token limit",
            "prompt is too long",
            "request too large",
            "input is too long",
            "413",
        )
        return any(marker in text for marker in markers)

    def _build_chunk_prompt(
        self,
        *,
        period_start: str,
        period_end: str,
        columns: Sequence[str],
        rows: List[Dict[str, Any]],
        batch_number: Optional[int] = None,
        total_batches: Optional[int] = None,
    ) -> str:
        # Determine display columns: put _source first if present
        has_source = any(row.get("_source") for row in rows)
        base_columns = [c for c in columns if c != "_source"]
        display_columns: List[str] = (["_source"] + base_columns) if has_source else list(base_columns)

        critical_rows = [row for row in rows if self._row_problem_score(row, list(columns)) > 0]
        extra_prompt_context = _read_prompt_setting("CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT")
        anti_rules = _resolve_anti_hallucination_rules()
        lower_cols = {str(c).lower() for c in display_columns}
        data_type = "aggregated" if {"start_time", "end_time", "cnt"}.issubset(lower_cols) else "raw"
        preferred_time = ("start_time", "timestamp", "time", "ts", "datetime", "end_time")
        time_column = next((c for c in preferred_time if c in lower_cols), "timestamp")
        if not self.prompt_context.get("data_type"):
            self.prompt_context["data_type"] = data_type
        if not self.prompt_context.get("time_column"):
            self.prompt_context["time_column"] = time_column

        source_stat = ""
        if has_source:
            source_counts: Dict[str, int] = {}
            for row in rows:
                src = str(row.get("_source") or "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            source_stat = ", ".join(
                f"{src}={cnt}" for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1])
            )

        log_lines: List[str] = []
        for idx, row in enumerate(rows, start=1):
            rendered_parts: List[str] = []
            for col in display_columns:
                value = row.get(col)
                if value is None or value == "":
                    continue
                text = self._truncate(str(value), self.config.max_cell_chars)
                rendered_parts.append(f"{col}={text}")
            log_lines.append(f"{idx}. " + " | ".join(rendered_parts))
        logs_text = "\n".join(log_lines) if log_lines else "Нет строк в батче."

        if bool(getattr(self.config, "use_new_algorithm", False)):
            user_context = _ctx_value(self.prompt_context, "incident_description", "")
            alerts_context = _ctx_value(self.prompt_context, "alerts_list", "")
            source_services = self._derive_source_services(rows)
            source_query = str(self.prompt_context.get("sql_query") or "").strip()
            batch_period_start, batch_period_end = _extract_batch_period(rows)
            map_batch_id = f"batch-{int(batch_number or 1):06d}"
            system_text = incident_prompts.MAP_SYSTEM_PROMPT.format(
                user_context=user_context,
                alerts_list=alerts_context,
                batch_id=map_batch_id,
                time_range_start=batch_period_start or period_start,
                time_range_end=batch_period_end or period_end,
                total_log_entries=len(rows),
                source_services=", ".join(source_services) if source_services else "unknown",
                source_query=source_query or "N/A",
            )
            user_text = incident_prompts.MAP_USER_PROMPT.format(
                batch_id=map_batch_id,
                log_entries=logs_text,
            )
            return f"{system_text}\n\n{user_text}"

        map_template = _read_prompt_setting("CONTROL_PLANE_LLM_MAP_PROMPT_TEMPLATE")
        if map_template:
            rendered = _render_prompt_template(
                map_template,
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "incident_start": _ctx_value(self.prompt_context, "incident_start", period_start),
                    "incident_end": _ctx_value(self.prompt_context, "incident_end", period_end),
                    "incident_description": _ctx_value(self.prompt_context, "incident_description", ""),
                    "alerts_list": _ctx_value(self.prompt_context, "alerts_list", ""),
                    "metrics_context": _ctx_value(self.prompt_context, "metrics_context", ""),
                    "source_name": _ctx_value(self.prompt_context, "source_name", source_stat or "query_1"),
                    "sql_query": _ctx_value(self.prompt_context, "sql_query", ""),
                    "batch_number": batch_number if batch_number is not None else "",
                    "total_batches": total_batches if total_batches is not None else "",
                    "batch_data": logs_text,
                    "rows_count": len(rows),
                    "problem_rows": len(critical_rows),
                    "columns": ", ".join(display_columns),
                    "time_column": _ctx_value(self.prompt_context, "time_column", time_column),
                    "data_type": _ctx_value(self.prompt_context, "data_type", data_type),
                    "source_distribution": source_stat,
                    "extra_prompt_context": extra_prompt_context,
                    "anti_hallucination_rules": anti_rules,
                    "logs_text": logs_text,
                },
            ).strip()
            return _append_chain_requirement(rendered, "map")

        lines = [
            "Это MAP-этап расследования инцидента. Анализируй только этот фрагмент логов.",
            "Если выше есть контекст алертов/инцидента — используй его как приоритет.",
            "",
            f"Источник: {_ctx_value(self.prompt_context, 'source_name', source_stat or 'query_1')}",
            f"SQL: {_ctx_value(self.prompt_context, 'sql_query', '')}",
            f"Батч: {batch_number if batch_number is not None else ''}/{total_batches if total_batches is not None else ''}",
            f"Период: [{period_start}, {period_end})",
            f"Поле времени: {time_column}",
            f"Тип данных: {data_type}",
            f"Строк в куске: {len(rows)}",
            f"Строк с problem-сигналами: {len(critical_rows)}",
            f"Колонки: {', '.join(display_columns)}",
        ]
        if source_stat:
            lines.append(f"Распределение по источникам: {source_stat}")
        if extra_prompt_context:
            lines += [
                "",
                "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДАННЫХ:",
                extra_prompt_context,
            ]
        lines += [
            "",
            "ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:",
            anti_rules,
            "",
            "Верни СТРОГО один JSON-объект (без markdown, без пояснений) для map-summary.",
            "ОБЯЗАТЕЛЬНЫЕ КЛЮЧИ JSON:",
            "context, timeline, causal_links, alert_refs, hypotheses, pinned_facts, gaps, impact, conflicts, data_quality, preliminary_recommendations",
            "Требования к JSON:",
            "- timeline[*].evidence_type: FACT или HYPOTHESIS",
            "- timeline[*].severity: critical|high|medium|low",
            "- importance/confidence/noise_ratio: число от 0 до 1",
            "- timestamp: ISO8601 с timezone и максимальной доступной точностью",
            "- все ссылки на события (event_id) должны существовать в timeline.id",
            "- для status в alert_refs используй: EXPLAINED|PARTIALLY|NOT_EXPLAINED|NOT_SEEN_IN_BATCH",
            "- для hypotheses.status используй: active|merged|conflicting|dismissed",
            "- для recommendations.priority используй: P0|P1|P2",
            "- НЕ фильтруй входные логи по типу (healthcheck/debug и т.п.): анализируй ровно переданный батч.",
            "",
            "Логи (хронологический порядок):",
            logs_text,
        ]
        return _append_chain_requirement("\n".join(lines), "map")

    def _build_reduce_prompt(
        self,
        *,
        period_start: str,
        period_end: str,
        reduce_round: int,
        summaries: List[str],
        sources: Optional[List[str]] = None,
    ) -> str:
        sources_line = (
            f"Источники данных: {', '.join(sources)}. "
            "Ищи причинно-следственные связи МЕЖДУ источниками."
            if sources else ""
        )
        extra_prompt_context = _read_prompt_setting("CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT")
        anti_rules = _resolve_anti_hallucination_rules()
        summaries_text = []
        for idx, text in enumerate(summaries, start=1):
            summaries_text.append(f"[BATCH {idx}]")
            summaries_text.append(text)
            summaries_text.append("")
        rendered_summaries = "\n".join(summaries_text).strip()

        if bool(getattr(self.config, "use_new_algorithm", False)):
            target_pct = max(min(int(getattr(self.config, "reduce_target_token_pct", 50) or 50), 90), 20)
            user_context = _ctx_value(self.prompt_context, "incident_description", "")
            alerts_context = _ctx_value(self.prompt_context, "alerts_list", "")
            summaries_payload: List[Any] = []
            for item in summaries:
                parsed_item = self._extract_json_payload(str(item or ""))
                summaries_payload.append(parsed_item if isinstance(parsed_item, dict) else str(item or ""))
            system_text = incident_prompts.REDUCE_SYSTEM_PROMPT.format(
                num_summaries=len(summaries),
                user_context=user_context,
                alerts_list=alerts_context,
                target_token_pct=target_pct,
            )
            user_text = incident_prompts.REDUCE_USER_PROMPT.format(
                num_summaries=len(summaries),
                summaries_json=json.dumps(summaries_payload, ensure_ascii=False),
            )
            return f"{system_text}\n\n{user_text}"

        reduce_template = _read_prompt_setting("CONTROL_PLANE_LLM_REDUCE_PROMPT_TEMPLATE")
        if reduce_template:
            rendered = _render_prompt_template(
                reduce_template,
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "incident_start": _ctx_value(self.prompt_context, "incident_start", period_start),
                    "incident_end": _ctx_value(self.prompt_context, "incident_end", period_end),
                    "incident_description": _ctx_value(self.prompt_context, "incident_description", ""),
                    "alerts_list": _ctx_value(self.prompt_context, "alerts_list", ""),
                    "metrics_context": _ctx_value(self.prompt_context, "metrics_context", ""),
                    "source_name": _ctx_value(self.prompt_context, "source_name", ", ".join(sources or [])),
                    "sql_query": _ctx_value(self.prompt_context, "sql_query", ""),
                    "data_type": _ctx_value(self.prompt_context, "data_type", ""),
                    "reduce_round": reduce_round,
                    "source_names": ", ".join(sources or []),
                    "sources_line": sources_line,
                    "batch_count": len(summaries),
                    "extra_prompt_context": extra_prompt_context,
                    "anti_hallucination_rules": anti_rules,
                    "summaries_text": rendered_summaries,
                    "map_summaries": json.dumps(summaries, ensure_ascii=False),
                    "map_summaries_text": rendered_summaries,
                },
            ).strip()
            return _append_chain_requirement(rendered, "reduce")

        lines = [
            "Это REDUCE-этап расследования инцидента.",
            "Если выше есть контекст инцидента/алертов — привяжи выводы к нему.",
            *(([sources_line]) if sources_line else []),
            "",
            f"Источник: {_ctx_value(self.prompt_context, 'source_name', ', '.join(sources or []))}",
            f"SQL: {_ctx_value(self.prompt_context, 'sql_query', '')}",
            f"Период: [{period_start}, {period_end})",
            f"Reduce round: {reduce_round}",
            f"Количество map-summary: {len(summaries)}",
        ]
        if extra_prompt_context:
            lines += [
                "",
                "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДАННЫХ:",
                extra_prompt_context,
            ]
        lines += [
            "",
            "ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:",
            anti_rules,
            "",
            "Объедини частичные summary в единый отчёт со строгими секциями:",
            "1) ОБЗОР ИСТОЧНИКА",
            "2) ХРОНОЛОГИЯ КЛЮЧЕВЫХ СОБЫТИЙ (только [РЕЛЕВАНТНО])",
            "3) ЦЕПОЧКА СОБЫТИЙ ИСТОЧНИКА (ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК, несколько цепочек допустимы)",
            "4) СВЯЗЬ МЕЖДУ ЦЕПОЧКАМИ: [СВЯЗАНЫ]/[ВОЗМОЖНО СВЯЗАНЫ]/[НЕЗАВИСИМЫ]",
            "5) ОБЪЯСНЕНИЕ АЛЕРТОВ",
            "6) ПЕРВОПРИЧИНЫ ПО ЦЕПОЧКАМ",
            "7) ПРОБЕЛЫ В ДАННЫХ И РАЗРЫВЫ ЦЕПОЧЕК",
            "8) ФОРМАТ ВЫВОДА ЦЕПОЧКИ: оформи красиво с узлами и стрелками.",
            "",
            "Частичные summary:",
            rendered_summaries,
        ]
        return _append_chain_requirement("\n".join(lines).strip(), "reduce")

    def _build_freeform_prompt(
        self,
        *,
        period_start: str,
        period_end: str,
        structured_summary: str,
        map_summaries: Optional[Sequence[str]] = None,
    ) -> str:
        extra_prompt_context = _read_prompt_setting("CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT")
        anti_rules = _resolve_anti_hallucination_rules()
        map_items = [_normalize_summary_text(item) for item in (map_summaries or [])]
        map_items = [item for item in map_items if item]
        map_summaries_text = "\n\n".join(
            f"[MAP SUMMARY #{idx + 1}]\n{item}" for idx, item in enumerate(map_items)
        )
        freeform_template = _read_prompt_setting("CONTROL_PLANE_LLM_FREEFORM_PROMPT_TEMPLATE")
        if freeform_template:
            rendered = _render_prompt_template(
                freeform_template,
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "incident_start": _ctx_value(self.prompt_context, "incident_start", period_start),
                    "incident_end": _ctx_value(self.prompt_context, "incident_end", period_end),
                    "incident_description": _ctx_value(self.prompt_context, "incident_description", ""),
                    "alerts_list": _ctx_value(self.prompt_context, "alerts_list", ""),
                    "metrics_context": _ctx_value(self.prompt_context, "metrics_context", ""),
                    "structured_summary": structured_summary,
                    "cross_source_summary": structured_summary,
                    "map_summaries": json.dumps(map_items, ensure_ascii=False),
                    "map_summaries_text": map_summaries_text,
                    "extra_prompt_context": extra_prompt_context,
                    "anti_hallucination_rules": anti_rules,
                },
            ).strip()
            return _append_chain_requirement(rendered, "freeform")
        return _append_chain_requirement("\n".join([
            "На основе структурированного анализа инцидента ниже напиши черновой нарратив.",
            "Это промежуточный результат для SRE-команды — 3-5 абзацев связным текстом.",
            "Включи: что произошло, в какой последовательности, вероятные причины с пометками [ФАКТ]/[ГИПОТЕЗА],",
            "что нужно проверить дополнительно.",
            "Пиши конкретно — ссылайся на реальные timestamp'ы и цитаты из логов, не генерируй абстракции.",
            "Если данных недостаточно для какого-то утверждения — прямо напиши об этом.",
            "Отдельным обязательным пунктом дай наглядную цепочку событий (узлы + стрелки).",
            "",
            "ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:",
            anti_rules,
            *((["", "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДАННЫХ:", extra_prompt_context]) if extra_prompt_context else []),
            *((["", "MAP SUMMARY ПО БАТЧАМ ЛОГОВ:", map_summaries_text]) if map_summaries_text else []),
            "",
            f"Период: [{period_start}, {period_end})",
            "",
            "Структурированный анализ:",
            structured_summary,
        ]), "freeform")

    def _rank_rows_by_problem_signal(
        self,
        rows: List[Dict[str, Any]],
        columns: Sequence[str],
    ) -> List[Dict[str, Any]]:
        return sorted(rows, key=lambda row: self._row_problem_score(row, columns), reverse=True)

    def _row_problem_score(self, row: Dict[str, Any], columns: Sequence[str]) -> int:
        score = 0
        text_parts = []
        for col in columns:
            value = row.get(col, "")
            if value is None:
                continue
            text_parts.append(str(value).lower())
        joined = " ".join(text_parts)
        for keyword in self.PROBLEM_KEYWORDS:
            if keyword in joined:
                score += 1
        if "level=error" in joined or "level=fatal" in joined:
            score += 2
        if "status=5" in joined or "http 5" in joined:
            score += 1
        return score

    @staticmethod
    def _validate_iso_datetime(value: str) -> None:
        datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _truncate(value: str, max_chars: int) -> str:
        if max_chars <= 0:
            return value
        if len(value) <= max_chars:
            return value
        if max_chars <= 3:
            return value[:max_chars]
        return value[: max_chars - 3] + "..."


def build_cross_source_reduce_prompt(
    summaries_by_source: Dict[str, str],
    period_start: str,
    period_end: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a cross-source REDUCE prompt that merges per-source MAP→REDUCE results.

    Used in the two-level summarization algorithm for multi-query mode:
      1. Per-source MAP→REDUCE (independent summaries per query).
      2. One cross-source REDUCE LLM call using this prompt.
    """
    ctx = context or {}
    sources = list(summaries_by_source.keys())
    source_blocks: List[str] = []
    for source, summary in summaries_by_source.items():
        source_blocks.append(f"=== {source} ===\n{summary}")
    source_summaries_text = "\n\n".join(source_blocks).strip()
    source_summaries_array = [
        {"source_name": str(source), "reduce_summary": str(summary)}
        for source, summary in summaries_by_source.items()
    ]
    anti_rules = _resolve_anti_hallucination_rules()
    extra_prompt_context = _read_prompt_setting("CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT")
    template = _read_prompt_setting("CONTROL_PLANE_LLM_CROSS_SOURCE_REDUCE_PROMPT_TEMPLATE")
    if template:
        rendered = _render_prompt_template(
            template,
            {
                "period_start": period_start,
                "period_end": period_end,
                "incident_start": _ctx_value(ctx, "incident_start", period_start),
                "incident_end": _ctx_value(ctx, "incident_end", period_end),
                "incident_description": _ctx_value(ctx, "incident_description", ""),
                "alerts_list": _ctx_value(ctx, "alerts_list", ""),
                "metrics_context": _ctx_value(ctx, "metrics_context", ""),
                "source_names": ", ".join(sources),
                "source_count": len(sources),
                "source_name": _ctx_value(ctx, "source_name", ", ".join(sources)),
                "sql_query": _ctx_value(ctx, "sql_query", ""),
                "time_column": _ctx_value(ctx, "time_column", ""),
                "data_type": _ctx_value(ctx, "data_type", ""),
                "source_summaries_text": source_summaries_text,
                "source_summaries": json.dumps(source_summaries_array, ensure_ascii=False),
                "extra_prompt_context": extra_prompt_context,
                "anti_hallucination_rules": anti_rules,
            },
        ).strip()
        return _append_chain_requirement(rendered, "cross")

    lines = [
        "Это финальный CROSS-SOURCE REDUCE: объедини результаты из нескольких источников.",
        "Привяжи выводы к контексту инцидента/алертов, если он указан выше.",
        f"Источники: {', '.join(sources)}.",
        "",
        f"Период: [{period_start}, {period_end})",
    ]
    if extra_prompt_context:
        lines += [
            "",
            "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДАННЫХ:",
            extra_prompt_context,
        ]
    lines += [
        "",
        "ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:",
        anti_rules,
        "",
        "Верни структурированный отчёт со секциями:",
        "1) ЕДИНАЯ ХРОНОЛОГИЯ ИНЦИДЕНТА",
        "2) КРОСС-КОРРЕЛЯЦИИ МЕЖДУ ИСТОЧНИКАМИ ([ФАКТ]/[ГИПОТЕЗА])",
        "3) ОБЪЯСНЕНИЕ АЛЕРТОВ (финальный вердикт)",
        "4) ЦЕПОЧКА СОБЫТИЙ ИНЦИДЕНТА (ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК)",
        "5) СВЯЗЬ МЕЖДУ ЦЕПОЧКАМИ: [ОДИН ИНЦИДЕНТ]/[ВОЗМОЖНО СВЯЗАНЫ]/[НЕЗАВИСИМЫ]",
        "6) ПЕРВОПРИЧИНЫ ПО ЦЕПОЧКАМ",
        "7) МАСШТАБ ВОЗДЕЙСТВИЯ",
        "8) РЕКОМЕНДАЦИИ ДЛЯ SRE (P0/P1/P2)",
        "9) ПРОБЕЛЫ И ОТКРЫТЫЕ ВОПРОСЫ",
        "",
        "Summary по источникам:",
        source_summaries_text,
    ]
    return _append_chain_requirement("\n".join(lines).strip(), "cross")


def _sections_to_markdown(sections: Sequence[InstructorFinalReportSection]) -> str:
    parts: List[str] = []
    for section in sections:
        title = str(getattr(section, "title", "") or "").strip() or "Untitled"
        text = _normalize_summary_text(getattr(section, "text", ""))
        parts.extend([f"## {title}", "", text or "Данных недостаточно.", ""])
    return "\n".join(parts).strip()


def _normalize_alert_status_for_report(status: str) -> str:
    raw = str(status or "").strip().upper()
    if raw == "EXPLAINED":
        return "EXPLAINED"
    if raw in {"PARTIALLY", "PARTIALLY_EXPLAINED"}:
        return "PARTIALLY_EXPLAINED"
    return "NOT_EXPLAINED"


def _render_alerts_list_text(alerts: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(alerts, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        details = str(item.get("details") or "").strip()
        time_mode = str(item.get("time_mode") or "").strip()
        timestamp = str(item.get("timestamp") or "").strip()
        start = str(item.get("start") or "").strip()
        end = str(item.get("end") or "").strip()
        head = title or f"alert_{idx}"
        payload = details if details else "(без деталей)"
        if time_mode == "range":
            time_block = f"{start} -> {end}".strip()
        else:
            time_block = timestamp
        if time_block:
            lines.append(f"{idx}. {head} | time={time_block} | {payload}")
        else:
            lines.append(f"{idx}. {head} | {payload}")
    return "\n".join(lines).strip()


def _build_empty_incident_summary(
    *,
    period_start: str,
    period_end: str,
) -> IncidentSummary:
    return IncidentSummary(
        context=IncidentContext(
            batch_id="final-report-input",
            time_range_start=period_start,
            time_range_end=period_end,
            total_log_entries=0,
            source_query=[],
            source_services=[],
        ),
        timeline=[],
        causal_links=[],
        alert_refs=[],
        hypotheses=[],
        pinned_facts=[],
        gaps=[],
        impact=IncidentImpact(),
        conflicts=[],
        data_quality=IncidentDataQuality(
            is_empty=True,
            noise_ratio=1.0,
            notes="IncidentSummary недоступен, использован пустой fallback.",
        ),
        preliminary_recommendations=[],
    )


def _report_from_summary_defaults(
    *,
    summary: IncidentSummary,
    user_goal: str,
    alerts: Sequence[Dict[str, Any]],
    metrics_context: str,
) -> IncidentReport:
    alert_ids_from_ui = [
        str(item.get("title") or f"alert_{idx + 1}").strip()
        for idx, item in enumerate(alerts)
        if isinstance(item, dict)
    ]
    alert_ids_from_ui = [item for item in alert_ids_from_ui if item]
    alert_by_id: Dict[str, IncidentAlertRef] = {
        str(item.alert_id): item for item in summary.alert_refs
    }
    all_alert_ids = list(dict.fromkeys(alert_ids_from_ui + list(alert_by_id.keys())))
    alert_explanations: List[AlertExplanation] = []
    for alert_id in all_alert_ids:
        matched = alert_by_id.get(alert_id)
        if matched is None:
            alert_explanations.append(
                AlertExplanation(
                    alert_id=alert_id,
                    status="NOT_EXPLAINED",
                    related_events=[],
                    explanation="В IncidentSummary нет прямой привязки этого алерта.",
                )
            )
            continue
        alert_explanations.append(
            AlertExplanation(
                alert_id=alert_id,
                status=_normalize_alert_status_for_report(matched.status),
                related_events=[str(eid) for eid in (matched.related_events or []) if str(eid).strip()],
                explanation=str(matched.explanation or "").strip(),
            )
        )

    chronology = [
        ChronologyEvent(
            id=str(item.id),
            timestamp=str(item.timestamp),
            source=str(item.source),
            description=str(item.description),
            severity=item.severity,
            evidence_type=item.evidence_type,
            evidence_quote=str(item.evidence_quote or "") if item.evidence_quote else None,
            tags=[str(tag) for tag in (item.tags or []) if str(tag).strip()],
        )
        for item in summary.timeline
    ]
    causal_chains = [
        CausalChain(
            id=str(item.id),
            cause_event_id=str(item.cause_event_id),
            effect_event_id=str(item.effect_event_id),
            mechanism=str(item.mechanism),
            confidence=float(item.confidence),
        )
        for item in summary.causal_links
    ]
    hypotheses = [
        ReportHypothesis(
            id=str(item.id),
            related_alert_ids=[str(aid) for aid in (item.related_alert_ids or []) if str(aid).strip()],
            title=str(item.title),
            description=str(item.description),
            confidence=float(item.confidence),
            confidence_rationale=(
                "Оценка confidence основана на количестве supporting/contradicting событий "
                "в текущем сводном IncidentSummary."
            ),
            supporting_events=[str(eid) for eid in (item.supporting_events or []) if str(eid).strip()],
            contradicting_events=[str(eid) for eid in (item.contradicting_events or []) if str(eid).strip()],
            status=item.status,
        )
        for item in summary.hypotheses
    ]
    conflicts = [
        ConflictDescription(
            id=str(item.id),
            description=str(item.description),
            side_a=str(item.side_a.description),
            side_b=str(item.side_b.description),
            side_a_events=[str(eid) for eid in (item.side_a.supporting_events or []) if str(eid).strip()],
            side_b_events=[str(eid) for eid in (item.side_b.supporting_events or []) if str(eid).strip()],
            resolution=str(item.resolution or "") or None,
        )
        for item in summary.conflicts
    ]
    gaps = [
        GapDescription(
            id=str(item.id),
            description=str(item.description),
            between_events=[str(eid) for eid in (item.between_events or []) if str(eid).strip()],
            missing_data=str(item.missing_data),
        )
        for item in summary.gaps
    ]
    impact_duration = ""
    if summary.impact.degradation_period is not None:
        impact_duration = (
            f"{summary.impact.degradation_period.start} -> "
            f"{summary.impact.degradation_period.end}"
        )
    impact = ImpactAssessment(
        affected_services=[str(item) for item in (summary.impact.affected_services or []) if str(item).strip()],
        affected_operations=[str(item) for item in (summary.impact.affected_operations or []) if str(item).strip()],
        error_counts=[str(item) for item in (summary.impact.error_counts or []) if str(item).strip()],
        duration=impact_duration,
        severity_assessment=(
            "high"
            if any(item.severity in {"critical", "high"} for item in summary.timeline)
            else "medium"
        ),
    )
    recommendations = [
        ReportRecommendation(
            id=str(item.id),
            priority=item.priority,
            action=str(item.action),
            rationale=str(item.rationale or ""),
            expected_effect="Снижение вероятности повторения инцидента.",
            related_hypothesis_ids=[str(hid) for hid in (item.related_hypothesis_ids or []) if str(hid).strip()],
        )
        for item in summary.preliminary_recommendations
    ]
    low_conf_hyp = [
        str(item.title)
        for item in summary.hypotheses
        if float(item.confidence) < 0.5
    ]
    max_conf = max((float(item.confidence) for item in summary.hypotheses), default=0.0)
    if max_conf >= 0.8:
        overall_confidence = "high"
    elif max_conf >= 0.5:
        overall_confidence = "medium"
    else:
        overall_confidence = "low"
    limitations = AnalysisLimitations(
        overall_confidence=overall_confidence,  # type: ignore[arg-type]
        rationale=(
            "Оценка построена на полноте timeline/causal_links/hypotheses и "
            "доступности объяснений по алертам."
        ),
        limitations=[
            item
            for item in [
                str(summary.data_quality.notes or "").strip(),
                *[
                    f"Gap {gap.id}: {gap.missing_data}"
                    for gap in gaps[:8]
                    if str(gap.missing_data).strip()
                ],
            ]
            if item
        ],
        low_confidence_hypotheses=low_conf_hyp,
    )
    metrics_provided = bool(str(metrics_context or "").strip())
    metrics = MetricsSection(
        metrics_provided=metrics_provided,
        anomalies=[],
        normal_metrics=[],
        recommendation_if_missing=(
            ""
            if metrics_provided
            else (
                "Метрики не предоставлены. Для более полного анализа добавьте: CPU, memory, "
                "latency(p95/p99), error rate, saturation по ключевым сервисам."
            )
        ),
    )
    summary_text = (
        "Инцидент проанализирован на основе верифицированного IncidentSummary. "
        f"Обработано событий: {len(chronology)}; гипотез: {len(hypotheses)}; "
        f"алертов в контексте: {len(all_alert_ids)}. "
        "Наиболее вероятная первопричина определяется по гипотезе с максимальным confidence."
    )
    if not chronology:
        summary_text = (
            "Значимые события в хронологии не обнаружены. "
            "Данных недостаточно для уверенного определения первопричины."
        )
    return IncidentReport(
        summary=ReportSummary(text=summary_text),
        data_coverage=DataCoverage(
            period_start=str(summary.context.time_range_start),
            period_end=str(summary.context.time_range_end),
            sql_queries=[str(item) for item in (summary.context.source_query or []) if str(item).strip()],
            services_covered=[str(item) for item in (summary.context.source_services or []) if str(item).strip()],
            services_missing=[],
            logs_processed=max(int(summary.context.total_log_entries), 0),
            notes=str(summary.data_quality.notes or ""),
        ),
        chronology=chronology,
        causal_chains=causal_chains,
        alert_explanations=alert_explanations,
        metrics=metrics,
        hypotheses=hypotheses,
        conflicts=conflicts,
        gaps=gaps,
        impact=impact,
        recommendations=recommendations,
        limitations=limitations,
    )


def _format_report_section_items(title: str, rows: Sequence[str]) -> str:
    cleaned = [str(item).strip() for item in rows if str(item).strip()]
    if not cleaned:
        return f"## {title}\n\nДанных недостаточно."
    return f"## {title}\n\n" + "\n".join(f"- {item}" for item in cleaned)


def _render_structured_sections_from_report(
    report: IncidentReport,
    *,
    user_context: str,
) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    sections.append(
        {
            "title": "1. Контекст Инцидента Из UI (Дословно)",
            "text": str(user_context or "").strip() or "Контекст инцидента в UI не задан.",
        }
    )
    sections.append({"title": "2. Резюме Инцидента", "text": str(report.summary.text or "").strip()})
    coverage_lines = [
        f"Период: {report.data_coverage.period_start} -> {report.data_coverage.period_end}",
        f"SQL: {', '.join(report.data_coverage.sql_queries) if report.data_coverage.sql_queries else 'n/a'}",
        f"Сервисы покрыты: {', '.join(report.data_coverage.services_covered) if report.data_coverage.services_covered else 'n/a'}",
        f"Сервисы не покрыты: {', '.join(report.data_coverage.services_missing) if report.data_coverage.services_missing else 'n/a'}",
        f"Обработано логов: {report.data_coverage.logs_processed}",
        f"Notes: {report.data_coverage.notes or 'n/a'}",
    ]
    sections.append({"title": "3. Покрытие Данных", "text": "\n".join(coverage_lines)})
    chronology_lines = [
        (
            f"[{item.timestamp}] ({item.source}) [{item.severity}] "
            f"[{item.evidence_type}] {item.description}"
            + (
                f" | quote: {item.evidence_quote}"
                if item.evidence_type == "FACT" and str(item.evidence_quote or "").strip()
                else ""
            )
        )
        for item in report.chronology
    ]
    sections.append({"title": "4. Полная Хронология Событий", "text": _format_report_section_items("4", chronology_lines).split("\n\n", 1)[1]})
    causal_lines = [
        (
            f"{item.cause_event_id} -> {item.effect_event_id} "
            f"(confidence={item.confidence:.2f}) | {item.mechanism}"
        )
        for item in report.causal_chains
    ]
    sections.append({"title": "5. Причинно-Следственные Цепочки", "text": _format_report_section_items("5", causal_lines).split("\n\n", 1)[1]})
    alert_lines = [
        (
            f"{item.alert_id}: {item.status} | events={', '.join(item.related_events) if item.related_events else 'n/a'}"
            + (f" | {item.explanation}" if str(item.explanation).strip() else "")
        )
        for item in report.alert_explanations
    ]
    sections.append({"title": "6. Связь С Каждым Инцидентом/Алертом Из UI", "text": _format_report_section_items("6", alert_lines).split("\n\n", 1)[1]})
    metrics_lines: List[str] = []
    if report.metrics.metrics_provided:
        for item in report.metrics.anomalies:
            metrics_lines.append(
                f"{item.metric_name}: {item.normal_value} -> {item.anomaly_value} | "
                f"{item.period_start}..{item.period_end} | {item.correlation_with_events}"
            )
        if report.metrics.normal_metrics:
            metrics_lines.append("Метрики в норме: " + ", ".join(report.metrics.normal_metrics))
    else:
        metrics_lines.append(
            report.metrics.recommendation_if_missing
            or "Метрики не предоставлены."
        )
    sections.append({"title": "7. Аномалии Метрик И Корреляции С Логами", "text": _format_report_section_items("7", metrics_lines).split("\n\n", 1)[1]})
    hyp_lines = [
        (
            f"{item.title} (confidence={item.confidence:.2f}, status={item.status}) | "
            f"alerts={', '.join(item.related_alert_ids) if item.related_alert_ids else 'n/a'} | "
            f"{item.description}"
        )
        for item in report.hypotheses
    ]
    sections.append({"title": "8. Гипотезы Первопричин", "text": _format_report_section_items("8", hyp_lines).split("\n\n", 1)[1]})
    conflict_lines = [
        f"{item.id}: {item.description} | A={item.side_a} | B={item.side_b}"
        + (f" | resolution={item.resolution}" if str(item.resolution or "").strip() else "")
        for item in report.conflicts
    ]
    sections.append({"title": "9. Конфликтующие Версии", "text": _format_report_section_items("9", conflict_lines).split("\n\n", 1)[1]})
    gap_lines = [
        f"{item.id}: {item.description} | between={', '.join(item.between_events) if item.between_events else 'n/a'} | need={item.missing_data}"
        for item in report.gaps
    ]
    sections.append({"title": "10. Разрывы В Цепочках", "text": _format_report_section_items("10", gap_lines).split("\n\n", 1)[1]})
    impact_lines = [
        f"Затронутые сервисы: {', '.join(report.impact.affected_services) if report.impact.affected_services else 'n/a'}",
        f"Затронутые операции: {', '.join(report.impact.affected_operations) if report.impact.affected_operations else 'n/a'}",
        f"Error counts: {', '.join(report.impact.error_counts) if report.impact.error_counts else 'n/a'}",
        f"Длительность: {report.impact.duration or 'n/a'}",
        f"Severity: {report.impact.severity_assessment or 'n/a'}",
    ]
    sections.append({"title": "11. Масштаб И Влияние", "text": "\n".join(impact_lines)})
    rec_lines = [
        (
            f"{item.priority}: {item.action}"
            + (f" | why={item.rationale}" if str(item.rationale).strip() else "")
            + (f" | effect={item.expected_effect}" if str(item.expected_effect).strip() else "")
        )
        for item in report.recommendations
    ]
    sections.append({"title": "12. Рекомендации Для SRE", "text": _format_report_section_items("12", rec_lines).split("\n\n", 1)[1]})
    limits_lines = [
        f"overall_confidence={report.limitations.overall_confidence}",
        f"rationale={report.limitations.rationale or 'n/a'}",
        (
            "limitations: "
            + (", ".join(report.limitations.limitations) if report.limitations.limitations else "n/a")
        ),
        (
            "low_confidence_hypotheses: "
            + (
                ", ".join(report.limitations.low_confidence_hypotheses)
                if report.limitations.low_confidence_hypotheses
                else "n/a"
            )
        ),
        report.limitations.note,
    ]
    sections.append({"title": "13. Уровень Уверенности И Ограничения Анализа", "text": "\n".join(limits_lines)})
    return sections


def _render_report_sections_to_markdown(sections: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    for item in sections:
        title = str(item.get("title") or "").strip() or "Untitled"
        text = _normalize_summary_text(item.get("text") or "")
        lines.extend([f"## {title}", "", text or "Данных недостаточно.", ""])
    return "\n".join(lines).strip()


def generate_final_reports_with_instructor(
    *,
    base_structured_report: str,
    base_freeform_report: str,
    user_goal: str,
    period_start: str,
    period_end: str,
    stats: Optional[Dict[str, Any]] = None,
    metrics_context: str = "",
    section_titles: Optional[Sequence[str]] = None,
    llm_timeout: float = 600.0,
    llm_max_retries: int = -1,
    model_supports_tool_calling: bool = True,
    verified_summary_json: str = "",
    alerts: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build final report through Instructor using 3-level fallback algorithm."""
    helper = PeriodLogSummarizer(
        db_fetch_page=lambda **_: [],
        llm_call=lambda _prompt: "",
        config=SummarizerConfig(
            use_instructor=True,
            model_supports_tool_calling=bool(model_supports_tool_calling),
        ),
        prompt_context={
            "llm_timeout": float(max(llm_timeout, 1.0)),
            "llm_max_retries": int(llm_max_retries),
        },
    )
    alerts_payload: List[Dict[str, Any]] = [
        dict(item) for item in (alerts or []) if isinstance(item, dict)
    ]
    alerts_text = _render_alerts_list_text(alerts_payload)
    user_context_text = _normalize_summary_text(user_goal) or "Контекст инцидента в UI не задан."
    source_summary_text = _normalize_summary_text(verified_summary_json) or _normalize_summary_text(base_structured_report)
    parsed_summary = helper._parse_incident_summary_text(
        source_summary_text,
        fallback_batch_id="final-report-input",
        fallback_start=period_start,
        fallback_end=period_end,
    )
    if parsed_summary is None:
        parsed_summary = _build_empty_incident_summary(period_start=period_start, period_end=period_end)
    summary_json = helper._incident_summary_to_json(parsed_summary)
    attempts_total = 0
    notes: List[str] = []
    stage_used = "attempt_1_full"
    report_obj: Optional[IncidentReport] = None

    full_prompt = "\n".join(
        [
            "Ты генерируешь финальный отчёт о расследовании инцидента для SRE-команды.",
            "Верни строго IncidentReport (пункты 2-13).",
            "Пункт 1 (контекст из UI) добавляется программно.",
            "summary (пункт 2) пиши последним и с полной картиной.",
            "Для каждого алерта из UI верни explanation со статусом EXPLAINED/PARTIALLY_EXPLAINED/NOT_EXPLAINED.",
            "Гипотезы обязательно привязывай к related_alert_ids.",
            "",
            "Контекст инцидента:",
            user_context_text,
            "",
            "Алерты из UI:",
            alerts_text or "n/a",
            "",
            f"Период: [{period_start}, {period_end})",
            f"Metrics context: {str(metrics_context or '').strip() or 'n/a'}",
            f"Stats: {json.dumps(stats or {}, ensure_ascii=False, default=str)}",
            "",
            "Верифицированное structured summary:",
            summary_json,
        ]
    )
    try:
        report_obj, attempts = helper._call_structured_with_instructor(
            prompt=full_prompt,
            response_model=IncidentReport,
            stage="final_report_full",
        )
        attempts_total += max(attempts, 1)
        notes.append("attempt_1_full=ok")
    except Exception as full_exc:  # noqa: BLE001
        notes.append(f"attempt_1_full=fail:{full_exc}")
        stage_used = "attempt_2_split"
        try:
            part1_prompt = "\n".join(
                [
                    "Сгенерируй ТОЛЬКО аналитические разделы отчёта в формате ReportPartAnalytical.",
                    "Нужны causal_chains, alert_explanations, hypotheses, recommendations, limitations.",
                    "",
                    "Контекст:",
                    user_context_text,
                    "",
                    "Алерты из UI:",
                    alerts_text or "n/a",
                    "",
                    "Structured summary:",
                    summary_json,
                ]
            )
            part1, attempts = helper._call_structured_with_instructor(
                prompt=part1_prompt,
                response_model=ReportPartAnalytical,
                stage="final_report_part1",
            )
            attempts_total += max(attempts, 1)
            part2_prompt = "\n".join(
                [
                    "Сгенерируй ТОЛЬКО описательные разделы отчёта в формате ReportPartDescriptive.",
                    "Используй аналитический контекст ниже.",
                    "",
                    "Контекст:",
                    user_context_text,
                    "",
                    "Алерты из UI:",
                    alerts_text or "n/a",
                    "",
                    "Structured summary:",
                    summary_json,
                    "",
                    "Analytical part JSON:",
                    json.dumps(part1.model_dump(mode='json'), ensure_ascii=False, indent=2),
                ]
            )
            part2, attempts = helper._call_structured_with_instructor(
                prompt=part2_prompt,
                response_model=ReportPartDescriptive,
                stage="final_report_part2",
            )
            attempts_total += max(attempts, 1)
            report_obj = IncidentReport(
                summary=part2.summary,
                data_coverage=part2.data_coverage,
                chronology=part2.chronology,
                causal_chains=part1.causal_chains,
                alert_explanations=part1.alert_explanations,
                metrics=part2.metrics,
                hypotheses=part1.hypotheses,
                conflicts=part2.conflicts,
                gaps=part2.gaps,
                impact=part2.impact,
                recommendations=part1.recommendations,
                limitations=part1.limitations,
            )
            notes.append("attempt_2_split=ok")
        except Exception as split_exc:  # noqa: BLE001
            notes.append(f"attempt_2_split=fail:{split_exc}")
            stage_used = "attempt_3_sectional"
            defaults = _report_from_summary_defaults(
                summary=parsed_summary,
                user_goal=user_context_text,
                alerts=alerts_payload,
                metrics_context=metrics_context,
            )

            def _section_call(
                *,
                prompt: str,
                response_model: Type[TModel],
                stage: str,
                trim_prompt: str = "",
            ) -> Optional[TModel]:
                nonlocal attempts_total
                try:
                    parsed, attempts = helper._call_structured_with_instructor(
                        prompt=prompt,
                        response_model=response_model,
                        stage=stage,
                    )
                    attempts_total += max(attempts, 1)
                    notes.append(f"{stage}=ok")
                    return parsed
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"{stage}=fail:{exc}")
                    if trim_prompt and _is_400_bad_request_exception(exc):
                        try:
                            parsed, attempts = helper._call_structured_with_instructor(
                                prompt=trim_prompt,
                                response_model=response_model,
                                stage=f"{stage}_trimmed",
                            )
                            attempts_total += max(attempts, 1)
                            notes.append(f"{stage}_trimmed=ok")
                            return parsed
                        except Exception as trim_exc:  # noqa: BLE001
                            notes.append(f"{stage}_trimmed=fail:{trim_exc}")
                    return None

            timeline_sorted = sorted(
                parsed_summary.timeline,
                key=lambda item: float(item.importance),
                reverse=True,
            )
            timeline_top = timeline_sorted[:20]
            chronology_prompt = (
                "Сформируй раздел chronology в формате ChronologySection.\n"
                "Сортировка строго по времени, укажи точные timestamps.\n\n"
                "Timeline JSON:\n"
                + json.dumps([item.model_dump(mode="json") for item in parsed_summary.timeline], ensure_ascii=False, indent=2)
            )
            chronology_trim_prompt = (
                "Сформируй раздел chronology в формате ChronologySection.\n"
                "Используй top events по importance.\n\n"
                "Timeline top JSON:\n"
                + json.dumps([item.model_dump(mode="json") for item in timeline_top], ensure_ascii=False, indent=2)
            )
            chronology_section = _section_call(
                prompt=chronology_prompt,
                response_model=ChronologySection,
                stage="section_chronology",
                trim_prompt=chronology_trim_prompt,
            )

            causal_prompt = (
                "Сформируй раздел causal_chains в формате CausalChainsSection.\n"
                "Нужен механизм связи для каждой цепочки.\n\n"
                "causal_links:\n"
                + json.dumps([item.model_dump(mode="json") for item in parsed_summary.causal_links], ensure_ascii=False, indent=2)
                + "\n\ncontext events:\n"
                + json.dumps([item.model_dump(mode="json") for item in timeline_top], ensure_ascii=False, indent=2)
            )
            causal_trim_prompt = (
                "Сформируй раздел causal_chains в формате CausalChainsSection из top-20 элементов.\n\n"
                + json.dumps(
                    [item.model_dump(mode="json") for item in parsed_summary.causal_links[:20]],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            causal_section = _section_call(
                prompt=causal_prompt,
                response_model=CausalChainsSection,
                stage="section_causal",
                trim_prompt=causal_trim_prompt,
            )

            alerts_prompt = (
                "Сформируй раздел alert_explanations в формате AlertsSection.\n"
                "Статусы: EXPLAINED/PARTIALLY_EXPLAINED/NOT_EXPLAINED.\n\n"
                "alerts from ui:\n"
                + (alerts_text or "n/a")
                + "\n\nalert_refs json:\n"
                + json.dumps([item.model_dump(mode="json") for item in parsed_summary.alert_refs], ensure_ascii=False, indent=2)
            )
            alerts_section = _section_call(
                prompt=alerts_prompt,
                response_model=AlertsSection,
                stage="section_alerts",
            )

            hypotheses_sorted = sorted(
                parsed_summary.hypotheses,
                key=lambda item: float(item.confidence),
                reverse=True,
            )
            hypotheses_prompt = (
                "Сформируй раздел hypotheses в формате HypothesesSection.\n"
                "Ранжируй по confidence, обязательно сохрани related_alert_ids.\n\n"
                "hypotheses:\n"
                + json.dumps([item.model_dump(mode="json") for item in parsed_summary.hypotheses], ensure_ascii=False, indent=2)
            )
            hypotheses_trim_prompt = (
                "Сформируй раздел hypotheses в формате HypothesesSection из top-20 гипотез.\n\n"
                + json.dumps([item.model_dump(mode="json") for item in hypotheses_sorted[:20]], ensure_ascii=False, indent=2)
            )
            hypotheses_section = _section_call(
                prompt=hypotheses_prompt,
                response_model=HypothesesSection,
                stage="section_hypotheses",
                trim_prompt=hypotheses_trim_prompt,
            )

            recommendations_prompt = (
                "Сформируй раздел recommendations в формате RecommendationsSection.\n"
                "Группируй P0/P1/P2, действия должны быть actionable.\n\n"
                "recommendations:\n"
                + json.dumps(
                    [item.model_dump(mode="json") for item in parsed_summary.preliminary_recommendations],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n\nhypotheses:\n"
                + json.dumps([item.model_dump(mode="json") for item in parsed_summary.hypotheses], ensure_ascii=False, indent=2)
            )
            recommendations_section = _section_call(
                prompt=recommendations_prompt,
                response_model=RecommendationsSection,
                stage="section_recommendations",
            )

            limitations_prompt = (
                "Сформируй раздел limitations в формате LimitationsSection.\n"
                "Нужна честная оценка ограничений и уверенности.\n\n"
                "data_quality:\n"
                + json.dumps(parsed_summary.data_quality.model_dump(mode="json"), ensure_ascii=False, indent=2)
                + "\n\ngaps:\n"
                + json.dumps([item.model_dump(mode="json") for item in parsed_summary.gaps], ensure_ascii=False, indent=2)
                + "\n\nhypotheses_confidence:\n"
                + json.dumps(
                    [{"id": item.id, "title": item.title, "confidence": item.confidence} for item in parsed_summary.hypotheses],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            limitations_section = _section_call(
                prompt=limitations_prompt,
                response_model=LimitationsSection,
                stage="section_limitations",
            )

            impact_prompt = (
                "Сформируй раздел impact в формате ImpactSection.\n\n"
                "impact json:\n"
                + json.dumps(parsed_summary.impact.model_dump(mode="json"), ensure_ascii=False, indent=2)
            )
            impact_section = _section_call(
                prompt=impact_prompt,
                response_model=ImpactSection,
                stage="section_impact",
            )

            gaps_section: Optional[GapsSection] = None
            if parsed_summary.gaps:
                gaps_prompt = (
                    "Сформируй раздел gaps в формате GapsSection.\n\n"
                    + json.dumps([item.model_dump(mode="json") for item in parsed_summary.gaps], ensure_ascii=False, indent=2)
                )
                gaps_section = _section_call(
                    prompt=gaps_prompt,
                    response_model=GapsSection,
                    stage="section_gaps",
                )

            conflicts_section: Optional[ConflictsSection] = None
            if parsed_summary.conflicts:
                conflicts_prompt = (
                    "Сформируй раздел conflicts в формате ConflictsSection.\n\n"
                    + json.dumps([item.model_dump(mode="json") for item in parsed_summary.conflicts], ensure_ascii=False, indent=2)
                )
                conflicts_section = _section_call(
                    prompt=conflicts_prompt,
                    response_model=ConflictsSection,
                    stage="section_conflicts",
                )

            coverage_prompt = (
                "Сформируй раздел data_coverage в формате CoverageSection.\n\n"
                "context:\n"
                + json.dumps(parsed_summary.context.model_dump(mode="json"), ensure_ascii=False, indent=2)
                + "\n\ndata_quality:\n"
                + json.dumps(parsed_summary.data_quality.model_dump(mode="json"), ensure_ascii=False, indent=2)
            )
            coverage_section = _section_call(
                prompt=coverage_prompt,
                response_model=CoverageSection,
                stage="section_coverage",
            )

            metrics_section: Optional[MetricsReportSection] = None
            if str(metrics_context or "").strip():
                metrics_prompt = (
                    "Сформируй раздел metrics в формате MetricsReportSection.\n\n"
                    f"metrics_context:\n{metrics_context}"
                )
                metrics_section = _section_call(
                    prompt=metrics_prompt,
                    response_model=MetricsReportSection,
                    stage="section_metrics",
                )

            assembled = IncidentReport(
                summary=defaults.summary,
                data_coverage=(
                    coverage_section.data_coverage
                    if coverage_section is not None
                    else defaults.data_coverage
                ),
                chronology=(
                    chronology_section.chronology
                    if chronology_section is not None
                    else defaults.chronology
                ),
                causal_chains=(
                    causal_section.causal_chains
                    if causal_section is not None
                    else defaults.causal_chains
                ),
                alert_explanations=(
                    alerts_section.alert_explanations
                    if alerts_section is not None
                    else defaults.alert_explanations
                ),
                metrics=(
                    metrics_section.metrics
                    if metrics_section is not None
                    else defaults.metrics
                ),
                hypotheses=(
                    hypotheses_section.hypotheses
                    if hypotheses_section is not None
                    else defaults.hypotheses
                ),
                conflicts=(
                    conflicts_section.conflicts
                    if conflicts_section is not None
                    else defaults.conflicts
                ),
                gaps=(
                    gaps_section.gaps
                    if gaps_section is not None
                    else defaults.gaps
                ),
                impact=(
                    impact_section.impact
                    if impact_section is not None
                    else defaults.impact
                ),
                recommendations=(
                    recommendations_section.recommendations
                    if recommendations_section is not None
                    else defaults.recommendations
                ),
                limitations=(
                    limitations_section.limitations
                    if limitations_section is not None
                    else defaults.limitations
                ),
            )
            summary_prompt = "\n".join(
                [
                    "Сформируй summary в формате SummarySection.",
                    "Summary должен быть написан ПОСЛЕДНИМ и учитывать все секции.",
                    "",
                    "sections_json:",
                    json.dumps(assembled.model_dump(mode="json"), ensure_ascii=False, indent=2),
                ]
            )
            summary_section = _section_call(
                prompt=summary_prompt,
                response_model=SummarySection,
                stage="section_summary_last",
            )
            if summary_section is not None:
                assembled = IncidentReport(
                    summary=summary_section.summary,
                    data_coverage=assembled.data_coverage,
                    chronology=assembled.chronology,
                    causal_chains=assembled.causal_chains,
                    alert_explanations=assembled.alert_explanations,
                    metrics=assembled.metrics,
                    hypotheses=assembled.hypotheses,
                    conflicts=assembled.conflicts,
                    gaps=assembled.gaps,
                    impact=assembled.impact,
                    recommendations=assembled.recommendations,
                    limitations=assembled.limitations,
                )
            report_obj = assembled
            notes.append("attempt_3_sectional=ok")

    if report_obj is None:
        report_obj = _report_from_summary_defaults(
            summary=parsed_summary,
            user_goal=user_context_text,
            alerts=alerts_payload,
            metrics_context=metrics_context,
        )
        notes.append("fallback=programmatic_defaults")

    structured_sections = _render_structured_sections_from_report(
        report_obj,
        user_context=user_context_text,
    )
    structured_text = _render_report_sections_to_markdown(structured_sections)

    freeform_text = ""
    freeform_section_items: List[Dict[str, str]] = []
    freeform_prompt = "\n".join(
        [
            "Напиши связный narrative-отчёт для SRE (1-2 страницы),",
            "используя структурированный IncidentReport ниже.",
            "Обязательно сохрани связь с алертами из UI и цепочки событий.",
            "",
            "Контекст инцидента (дословно):",
            user_context_text,
            "",
            "IncidentReport JSON:",
            json.dumps(report_obj.model_dump(mode="json"), ensure_ascii=False, indent=2),
        ]
    )
    try:
        freeform_obj, attempts = helper._call_structured_with_instructor(
            prompt=freeform_prompt,
            response_model=FreeformReport,
            stage="final_report_freeform",
        )
        attempts_total += max(attempts, 1)
        freeform_text = _normalize_summary_text(freeform_obj.text)
        notes.append("freeform=ok")
    except Exception as freeform_exc:  # noqa: BLE001
        notes.append(f"freeform=fail:{freeform_exc}")
        freeform_text = _normalize_summary_text(base_freeform_report)
    if not freeform_text:
        freeform_text = (
            "Краткий narrative-отчёт:\n\n"
            + _normalize_summary_text(report_obj.summary.text)
            + "\n\nДетали и доказательства смотрите в структурированном отчёте."
        )
    freeform_section_items = [{"title": "Narrative", "text": freeform_text}]

    return {
        "structured_report": structured_text,
        "freeform_report": freeform_text,
        "structured_sections": structured_sections,
        "freeform_sections": freeform_section_items,
        "notes": "; ".join([item for item in notes if item]),
        "attempts": int(max(attempts_total, 1)),
        "algorithm_stage": stage_used,
        "incident_report": report_obj.model_dump(mode="json"),
    }


def regenerate_reduce_summary_from_map_summaries(
    *,
    map_summaries: Sequence[str],
    period_start: str,
    period_end: str,
    llm_call: LLMTextCaller,
    prompt_context: Optional[Dict[str, Any]] = None,
    on_progress: Optional[ProgressCallback] = None,
    config: Optional[SummarizerConfig] = None,
) -> str:
    """
    Rebuild final REDUCE summary from already prepared MAP summaries.
    Useful for "rerun final summary" without refetching logs.
    """
    prepared = [_normalize_summary_text(item) for item in map_summaries]
    prepared = [item for item in prepared if item and not _is_llm_error_stub(item)]
    if not prepared:
        return "Нет map-summary для повторного REDUCE."

    reducer = PeriodLogSummarizer(
        db_fetch_page=lambda **_: [],
        llm_call=llm_call,
        config=config
        or SummarizerConfig(
            min_llm_chunk_rows=max(
                int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MIN_LLM_BATCH_SIZE", 20) or 20),
                1,
            ),
            auto_shrink_on_400=bool(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_AUTO_SHRINK_ON_400", True)
            ),
            max_shrink_rounds=max(
                int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SHRINK_ROUNDS", 6) or 6),
                0,
            ),
            max_cell_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS", 0)
            ),
            max_summary_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS", 0)
            ),
            reduce_prompt_max_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS", 0)
            ),
            use_instructor=bool(getattr(settings, "CONTROL_PLANE_LLM_USE_INSTRUCTOR", True)),
            model_supports_tool_calling=bool(
                getattr(settings, "CONTROL_PLANE_LLM_SUPPORTS_TOOL_CALLING", True)
            ),
        ),
        on_progress=on_progress,
        prompt_context=prompt_context or {},
    )
    final_summary, _, _ = reducer._reduce_summaries(
        chunk_summaries=list(prepared),
        period_start=period_start,
        period_end=period_end,
        sources=None,
    )
    return _normalize_summary_text(final_summary) or "Пустой итог повторного REDUCE."


def summarize_logs(
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    anomaly: Optional[Dict[str, Any]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    LLM map-reduce summarizer over paged logs from ClickHouse.
    Can be wired as CONTROL_PLANE_SUMMARIZER_CALLABLE=my_summarizer.summarize_logs
    """
    start_iso, end_iso = _normalize_period(
        period_start=period_start,
        period_end=period_end,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    fetch_mode = _resolve_logs_fetch_mode()
    tail_limit = max(int(getattr(settings, "CONTROL_PLANE_LOGS_TAIL_LIMIT", 1000)), 1)
    service = _resolve_service(anomaly)
    page_limit = int(settings.CONTROL_PLANE_LOGS_PAGE_LIMIT)
    total_rows_estimate = _estimate_total_logs(
        anomaly=anomaly,
        period_start=start_iso,
        period_end=end_iso,
        page_limit=page_limit,
        fetch_mode=fetch_mode,
        tail_limit=tail_limit,
    )

    fetch_errors: List[str] = []

    def _on_fetch_error(msg: str) -> None:
        fetch_errors.append(msg)
        if on_progress:
            on_progress("fetch_error", {"error": msg})

    db_fetch_page = _build_db_fetch_page(
        anomaly,
        fetch_mode=fetch_mode,
        tail_limit=tail_limit,
        on_error=_on_fetch_error,
    )
    llm_call = _make_llm_call()
    summarizer = PeriodLogSummarizer(
        db_fetch_page=db_fetch_page,
        llm_call=llm_call,
        config=SummarizerConfig(
            page_limit=page_limit,
            llm_chunk_rows=max(
                int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE", 200) or 200),
                1,
            ),
            min_llm_chunk_rows=max(
                int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MIN_LLM_BATCH_SIZE", 20) or 20),
                1,
            ),
            auto_shrink_on_400=bool(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_AUTO_SHRINK_ON_400", True)
            ),
            max_shrink_rounds=max(
                int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SHRINK_ROUNDS", 6) or 6),
                0,
            ),
            max_cell_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS", 0)
            ),
            max_summary_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS", 0)
            ),
            reduce_prompt_max_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS", 0)
            ),
            use_instructor=bool(getattr(settings, "CONTROL_PLANE_LLM_USE_INSTRUCTOR", True)),
            model_supports_tool_calling=bool(
                getattr(settings, "CONTROL_PLANE_LLM_SUPPORTS_TOOL_CALLING", True)
            ),
        ),
        on_progress=on_progress,
        prompt_context={
            "incident_start": start_iso,
            "incident_end": end_iso,
            "incident_description": "",
            "alerts_list": "",
            "metrics_context": "",
            "source_name": service,
            "sql_query": _resolve_logs_query_template(),
            "time_column": str(getattr(settings, "CONTROL_PLANE_LOGS_TIMESTAMP_COLUMN", "timestamp")),
            "data_type": "",
        },
    )
    result = summarizer.summarize_period(
        period_start=start_iso,
        period_end=end_iso,
        columns=list(DEFAULT_SUMMARY_COLUMNS),
        total_rows_estimate=total_rows_estimate,
    )
    summary_text = str(result.summary)
    if summary_text and not summary_text.startswith("Сервис:"):
        summary_text = f"Сервис: {service}. {summary_text}"

    return {
        "summary": summary_text,
        "freeform_summary": result.freeform_summary,
        "fetch_errors": fetch_errors,
        "chunk_summaries": result.map_summaries,
        "map_batches": result.map_batches,
        "pages_fetched": result.pages_fetched,
        "rows_processed": result.rows_processed,
        "llm_calls": result.llm_calls,
        "reduce_rounds": result.reduce_rounds,
        "rows_total_estimate": total_rows_estimate,
        "logs_fetch_mode": fetch_mode,
        "logs_tail_limit": tail_limit,
        "source": "llm_map_reduce",
    }
