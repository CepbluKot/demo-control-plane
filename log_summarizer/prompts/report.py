"""Промпты для финального отчёта."""

from __future__ import annotations

from typing import Optional

# ── Системные промпты ──────────────────────────────────────────────────

_REPORT_SYSTEM_FULL = """\
You are a senior SRE writing a post-incident analysis report.

Write a clear, structured Markdown report. Be specific: use timestamps, service names,
exact error messages from the evidence. Avoid vague phrases like "an issue occurred".

Structure:
## Executive Summary
1-2 paragraphs: what happened, when, impact.

## Alert Coverage
For each alert in the alert_refs section: its ID, name, status (explained/partial/not_explained/not_seen),
and a brief explanation. If no alerts provided, omit this section.

## Timeline
Chronological bullet list of key events with timestamps and sources.
Mark high-importance events (importance ≥ 0.8) with ⚠.

## Root Cause Analysis
What caused the incident. Reference evidence and causal chains.
Use sub-sections if multiple contributing factors.

## Impact
Which services were affected, for how long, estimated user/business impact.

## Evidence
Verbatim log lines supporting the root cause (from evidence_bank).
Format: `[timestamp] [source] raw_line`

## Contributing Factors & Anomalies
What else was going on that contributed or was unusual.

## Hypotheses
List each hypothesis with confidence level and supporting/contradicting evidence.
Include related alerts where applicable.

## Recommendations
Concrete action items to prevent recurrence or reduce impact next time.
Include preliminary_recommendations from the analysis if present.

Rules:
- Use exact timestamps from the data
- Quote evidence verbatim in code blocks
- Keep each section focused; no padding
- If information is missing, say so explicitly rather than speculating
"""

# Секционные промпты — для split-режима когда полный промпт не влезает

_REPORT_SECTION_ANALYSIS = """\
You are a senior SRE writing part of a post-incident report.

Write ONLY the following sections in Markdown:
## Executive Summary
## Alert Coverage
## Timeline
## Root Cause Analysis
## Impact

Use the analysis JSON provided. Be specific: timestamps, service names, causal chains.
For Alert Coverage: list each alert with its status and brief explanation (omit section if no alerts).
For Timeline: mark high-importance events (importance ≥ 0.8) with ⚠.
Do not add other sections. No preamble.
"""

_REPORT_SECTION_EVIDENCE = """\
You are a senior SRE writing part of a post-incident report.

Write ONLY the following section in Markdown:
## Evidence

List verbatim log lines from the evidence_bank that support the root cause.
Format each line as:
`[timestamp] [source] raw_line`

Group by theme if there are many lines. No other sections. No preamble.
"""

_REPORT_SECTION_RECOMMENDATIONS = """\
You are a senior SRE writing part of a post-incident report.

Write ONLY the following sections in Markdown:
## Contributing Factors & Anomalies
## Hypotheses
## Recommendations

Use the analysis JSON provided. For Recommendations: concrete action items only,
no vague advice. Reference specific services/components by name.

No other sections. No preamble.
"""


# ── Публичный API ──────────────────────────────────────────────────────

def build_report_system_prompt(section: Optional[str] = None) -> str:
    """Возвращает системный промпт для отчёта.

    Args:
        section: None = полный отчёт; "analysis" / "evidence" / "recommendations"
                 = один из трёх секционных промптов для split-режима.
    """
    if section is None:
        return _REPORT_SYSTEM_FULL
    if section == "analysis":
        return _REPORT_SECTION_ANALYSIS
    if section == "evidence":
        return _REPORT_SECTION_EVIDENCE
    if section == "recommendations":
        return _REPORT_SECTION_RECOMMENDATIONS
    raise ValueError(f"Unknown report section: {section!r}")


def format_report_user_prompt(
    analysis_json: str,
    evidence_text: str,
    early_summaries_text: str,
    incident_context: str,
    incident_start: str,
    incident_end: str,
    alert_refs_text: str = "",
) -> str:
    """Форматирует user-промпт для финального отчёта (полного или секционного).

    Args:
        analysis_json: Сериализованный MergedAnalysis (без evidence_bank/alert_refs).
        evidence_text: Конкатенация evidence_bank строк.
        early_summaries_text: Ранние промежуточные саммари.
        incident_context: Описание инцидента.
        incident_start: ISO8601 строка начала периода.
        incident_end: ISO8601 строка конца периода.
        alert_refs_text: Статусы алертов (программный merge из MAP-фазы).
    """
    parts: list[str] = [
        f"## Incident context\n{incident_context}",
        f"Period: {incident_start} → {incident_end}",
        "",
        "## Merged analysis (JSON)",
        "```json",
        analysis_json,
        "```",
    ]

    if alert_refs_text.strip():
        parts += [
            "",
            "## Alert coverage (status per alert)",
            alert_refs_text,
        ]

    if evidence_text.strip():
        parts += [
            "",
            "## Evidence bank (verbatim log lines)",
            "```",
            evidence_text,
            "```",
        ]

    if early_summaries_text.strip():
        parts += [
            "",
            "## Early-phase summaries (additional context)",
            early_summaries_text,
        ]

    return "\n".join(parts)
