"""Dify Code Node: MAP — анализ одного батча логов.

Inputs:
  item          (str) — батч логов
  incident_info (str)
  period_start  (str)
  period_end    (str)
  llm_api_base  (str)
  llm_api_key   (str)
  llm_model     (str)
  llm_timeout   (str) — default "300"
Outputs: analysis (Object)
"""
import json
import re
import ssl
import urllib.request


def main(
    item: str,
    incident_info: str,
    period_start: str,
    period_end: str,
    llm_api_base: str,
    llm_api_key: str,
    llm_model: str,
    llm_timeout: str = "300",
) -> dict:
    base = llm_api_base.rstrip("/")
    url = (base if base.endswith("/v1") else base + "/v1") + "/chat/completions"

    system = f"""You are a senior SRE analyzing a production log batch during an active incident.

=== Incident context ===
{incident_info}

Period: {period_start} → {period_end}

=== Task ===
Analyze the log batch and output a single JSON object with this structure:
{{
  "time_range": ["<ISO8601>", "<ISO8601>"],
  "narrative": "<what happened in this batch>",
  "events": [{{"id": "evt-1", "timestamp": "<ISO>", "source": "<pod>", "description": "<text>", "severity": "critical|high|medium|low|info", "importance": 0.0, "tags": []}}],
  "evidence": [{{"id": "ev-1", "timestamp": "<ISO>", "source": "<pod>", "raw_line": "<exact log line>", "severity": "info", "linked_event_id": null}}],
  "hypotheses": [{{"id": "hyp-1", "title": "<title>", "description": "<text>", "reasoning": "<text>", "confidence": "low|medium|high", "supporting_event_ids": [], "contradicting_event_ids": [], "related_alert_ids": []}}],
  "anomalies": [],
  "preliminary_recommendations": []
}}

Output ONLY the JSON object, nothing else."""

    user = f"## Log batch {period_start} → {period_end}\n\n```\n{item}\n```"

    payload = json.dumps({
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {llm_api_key}")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, timeout=float(llm_timeout), context=ctx) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        raw = result["choices"][0]["message"]["content"]

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return {"analysis": json.loads(raw)}
