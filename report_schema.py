from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReportSummary(BaseModel):
    text: str = Field(
        description="Краткое резюме инцидента (3-5 предложений)",
    )


class DataCoverage(BaseModel):
    period_start: str = ""
    period_end: str = ""
    sql_queries: list[str] = Field(default_factory=list)
    services_covered: list[str] = Field(default_factory=list)
    services_missing: list[str] = Field(default_factory=list)
    logs_processed: int = 0
    notes: str = ""


class ChronologyEvent(BaseModel):
    id: str
    timestamp: str
    source: str
    description: str
    severity: Literal["critical", "high", "medium", "low"]
    evidence_type: Literal["FACT", "HYPOTHESIS"]
    evidence_quote: str | None = None
    tags: list[str] = Field(default_factory=list)


class CausalChain(BaseModel):
    id: str
    cause_event_id: str
    effect_event_id: str
    mechanism: str
    confidence: float = Field(ge=0.0, le=1.0)


class AlertExplanation(BaseModel):
    alert_id: str
    status: Literal["EXPLAINED", "PARTIALLY_EXPLAINED", "NOT_EXPLAINED"]
    related_events: list[str] = Field(default_factory=list)
    explanation: str = ""


class MetricAnomaly(BaseModel):
    metric_name: str
    normal_value: str = ""
    anomaly_value: str = ""
    period_start: str = ""
    period_end: str = ""
    correlation_with_events: str = ""


class MetricsSection(BaseModel):
    metrics_provided: bool = False
    anomalies: list[MetricAnomaly] = Field(default_factory=list)
    normal_metrics: list[str] = Field(default_factory=list)
    recommendation_if_missing: str = ""


class ReportHypothesis(BaseModel):
    id: str
    related_alert_ids: list[str] = Field(default_factory=list)
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_rationale: str = ""
    supporting_events: list[str] = Field(default_factory=list)
    contradicting_events: list[str] = Field(default_factory=list)
    status: Literal["active", "merged", "conflicting", "dismissed"] = "active"


class ConflictDescription(BaseModel):
    id: str
    description: str
    side_a: str = ""
    side_b: str = ""
    side_a_events: list[str] = Field(default_factory=list)
    side_b_events: list[str] = Field(default_factory=list)
    resolution: str | None = None


class GapDescription(BaseModel):
    id: str
    description: str
    between_events: list[str] = Field(default_factory=list)
    missing_data: str = ""


class ImpactAssessment(BaseModel):
    affected_services: list[str] = Field(default_factory=list)
    affected_operations: list[str] = Field(default_factory=list)
    error_counts: list[str] = Field(default_factory=list)
    duration: str = ""
    severity_assessment: str = ""


class ReportRecommendation(BaseModel):
    id: str
    priority: Literal["P0", "P1", "P2"]
    action: str
    rationale: str = ""
    expected_effect: str = ""
    related_hypothesis_ids: list[str] = Field(default_factory=list)


class AnalysisLimitations(BaseModel):
    overall_confidence: Literal["high", "medium", "low"] = "medium"
    rationale: str = ""
    limitations: list[str] = Field(default_factory=list)
    low_confidence_hypotheses: list[str] = Field(default_factory=list)
    note: str = (
        "Этот отчёт — помощь в расследовании, не замена инженерного суждения SRE."
    )


class IncidentReport(BaseModel):
    summary: ReportSummary
    data_coverage: DataCoverage
    chronology: list[ChronologyEvent] = Field(default_factory=list)
    causal_chains: list[CausalChain] = Field(default_factory=list)
    alert_explanations: list[AlertExplanation] = Field(default_factory=list)
    metrics: MetricsSection = Field(default_factory=MetricsSection)
    hypotheses: list[ReportHypothesis] = Field(default_factory=list)
    conflicts: list[ConflictDescription] = Field(default_factory=list)
    gaps: list[GapDescription] = Field(default_factory=list)
    impact: ImpactAssessment = Field(default_factory=ImpactAssessment)
    recommendations: list[ReportRecommendation] = Field(default_factory=list)
    limitations: AnalysisLimitations = Field(default_factory=AnalysisLimitations)


class ReportPartAnalytical(BaseModel):
    causal_chains: list[CausalChain] = Field(default_factory=list)
    alert_explanations: list[AlertExplanation] = Field(default_factory=list)
    hypotheses: list[ReportHypothesis] = Field(default_factory=list)
    recommendations: list[ReportRecommendation] = Field(default_factory=list)
    limitations: AnalysisLimitations = Field(default_factory=AnalysisLimitations)


class ReportPartDescriptive(BaseModel):
    summary: ReportSummary
    data_coverage: DataCoverage
    chronology: list[ChronologyEvent] = Field(default_factory=list)
    metrics: MetricsSection = Field(default_factory=MetricsSection)
    conflicts: list[ConflictDescription] = Field(default_factory=list)
    gaps: list[GapDescription] = Field(default_factory=list)
    impact: ImpactAssessment = Field(default_factory=ImpactAssessment)


class ChronologySection(BaseModel):
    chronology: list[ChronologyEvent] = Field(default_factory=list)


class CausalChainsSection(BaseModel):
    causal_chains: list[CausalChain] = Field(default_factory=list)


class AlertsSection(BaseModel):
    alert_explanations: list[AlertExplanation] = Field(default_factory=list)


class HypothesesSection(BaseModel):
    hypotheses: list[ReportHypothesis] = Field(default_factory=list)


class RecommendationsSection(BaseModel):
    recommendations: list[ReportRecommendation] = Field(default_factory=list)


class LimitationsSection(BaseModel):
    limitations: AnalysisLimitations = Field(default_factory=AnalysisLimitations)


class ImpactSection(BaseModel):
    impact: ImpactAssessment = Field(default_factory=ImpactAssessment)


class GapsSection(BaseModel):
    gaps: list[GapDescription] = Field(default_factory=list)


class ConflictsSection(BaseModel):
    conflicts: list[ConflictDescription] = Field(default_factory=list)


class CoverageSection(BaseModel):
    data_coverage: DataCoverage


class MetricsReportSection(BaseModel):
    metrics: MetricsSection = Field(default_factory=MetricsSection)


class SummarySection(BaseModel):
    summary: ReportSummary


class FreeformReport(BaseModel):
    text: str
