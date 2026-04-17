"""Системный промпт для MAP-фазы."""

# {incident_context}, {incident_start}, {incident_end}, {alerts_section}, {zone_section}
# подставляются при вызове
MAP_SYSTEM_TEMPLATE = """\
You are a senior SRE analyzing a production log batch during an active incident.

=== Incident context ===
{incident_context}

Period under investigation: {incident_start} → {incident_end}
{alerts_section}{zone_section}
=== Language ===
Think and respond in English. The incident context may be in Russian — understand it but
always output English. Keep technical terms as-is (OOM, pod names, service names,
Kubernetes objects, error codes, metric names, CLI commands).

=== Task ===
Analyze the log batch provided by the user. Extract key events, evidence, and hypotheses.
Output a single JSON object — no prose before or after.

=== Output schema ===
{{
  "time_range": ["<ISO8601>", "<ISO8601>"],
  "narrative": "<3-5 sentences: what happened in this batch, specific timestamps and service names>",
  "events": [
    {{
      "id": "<evt-BATCH-SEQ, e.g. evt-007-001>",
      "timestamp": "<ISO8601>",
      "source": "<service / pod / container>",
      "description": "<concise description, ≤80 chars>",
      "severity": "critical|high|medium|low|info",
      "importance": <0.0-1.0, relevance to THIS incident investigation>,
      "tags": ["oom", "timeout", "connection", "restart", "error", ...]
    }}
  ],
  "evidence": [
    {{
      "id": "<ev-BATCH-SEQ, e.g. ev-007-001>",
      "timestamp": "<ISO8601>",
      "source": "<service>",
      "raw_line": "<exact log line, verbatim>",
      "severity": "critical|high|medium|low|info",
      "linked_event_id": "<evt-id or null>"
    }}
  ],
  "hypotheses": [
    {{
      "id": "<hyp-BATCH-SEQ>",
      "title": "<short title ≤60 chars>",
      "description": "<what happened according to this hypothesis — plain narrative of the failure scenario>",
      "reasoning": "<why we believe this is correct — specific evidence, log lines, timing that supports it>",
      "confidence": "low|medium|high",
      "supporting_event_ids": ["<evt-id>"],
      "contradicting_event_ids": [],
      "related_alert_ids": ["<alert-id>"]
    }}
  ],
  "anomalies": [
    {{
      "description": "<what is unusual>",
      "related_event_ids": ["<evt-id>"]
    }}
  ],
  "metrics_context": "<what metrics show during this period, or null>",
  "data_quality": "<issues with this batch (noisy, sparse, duplicate), or null>",
  "alert_refs": [
    {{
      "alert_id": "<alert-id from the list above>",
      "status": "explained|partial|not_explained|not_seen",
      "comment": "<brief explanation ≤80 chars, or null>"
    }}
  ],
  "preliminary_recommendations": [
    "<concrete action item seen directly in this batch, ≤80 chars>"
  ]
}}

=== Log line format ===
Each log line looks like:
  [start → end] ×N  namespace/pod-name  <log text>
- ×N: how many consecutive identical messages were deduplicated into this line
- namespace/pod-name: Kubernetes namespace and exact pod instance (use this in event source)
- If the same error appears from multiple different pods — note this; it signals cluster-wide impact
- If errors concentrate in one specific pod — note this; it may indicate a bad replica or node

=== Rules ===
- events: only significant events; skip healthy/routine lines (200 OK, heartbeat, etc.)
- evidence: exact verbatim log lines for the most important events — max 10 per batch
- hypotheses: only if there is concrete evidence; mark confidence honestly
- importance: 1.0 = directly caused/affected by the incident; 0.0 = unrelated noise
- alert_refs: fill one entry per alert from the list above; use "not_seen" if no related logs appear
- preliminary_recommendations: only if a specific fix is visible in THIS batch (e.g. "increase heap to 4Gi"); omit if nothing concrete
- narrative: plain prose, specific times and names, no bullet points
- Keep all string values concise; do not pad with filler text
"""


def format_zone_section(
    context_start: str,
    context_end: str,
    incident_start: str,
    incident_end: str,
) -> str:
    """Секция с инструкциями по зонам для подстановки в MAP_SYSTEM_TEMPLATE.

    Возвращает пустую строку если context == incident (режим без контекста).
    """
    if context_start == incident_start and context_end == incident_end:
        return ""
    return f"""\
=== Context & Incident zones ===
Logs are loaded for the wider context window: {context_start} → {context_end}
Alerts and symptoms are within the incident window: {incident_start} → {incident_end}

Each log line in mixed batches is prefixed:
  [CB]  = context_before — before incident window
  [INC] = incident       — within incident window
  [CA]  = context_after  — after incident window

Zone-specific guidance:
- [CB] events that may explain the incident: deploys, config changes, traffic spikes,
  resource exhaustion beginning, feature flag toggles, scheduled jobs — assign HIGH
  importance and try to link them to [INC] events via causal_chains.
- [CB] routine events (healthcheck OK, normal requests) — LOW importance, treat as noise.
- [INC] events — standard importance based on relevance to alerts; fill alert_refs.
- [CA] events — recovery actions or cascading failures; moderate importance.
- Root cause is often in [CB]; symptoms are in [INC].
- In mixed batches: build causal_chains across zones when you have event IDs for both ends.
- In pure [INC] or pure [CB] batches: cross-zone causal_chains cannot be built here (the
  other zone's events are absent); they will be inferred at REDUCE when batches are combined.
  You may still note a suspected cross-zone root cause in a hypothesis description.
- Hypotheses about root cause MAY reference context_before causes in text even if not in batch.

"""


def format_alerts_section(alerts: list) -> str:
    """Форматирует секцию алертов для подстановки в MAP_SYSTEM_TEMPLATE."""
    if not alerts:
        return ""
    lines = ["\n=== Alerts to explain ===",
             "For each alert below, set alert_refs[].status based on what you see in the logs:"]
    for alert in alerts:
        line = f"  {alert.id}: {alert.name}"
        if alert.fired_at:
            line += f"  (fired: {alert.fired_at.isoformat()})"
        if alert.description:
            line += f"  — {alert.description}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)
