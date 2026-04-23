"""Dify Code Node: MAP — анализ одного батча логов.

Ставится внутри Iteration Node вместо LLM Node.
Inputs:
  item          (str) — батч логов (JSON-строка с массивом записей)
  incident_info (str) — описание инцидента
  period_start  (str) — начало периода, ISO 8601
  period_end    (str) — конец периода, ISO 8601
  llm_api_base  (str) — endpoint LLM (без /v1)
  llm_api_key   (str) — API ключ
  llm_model     (str) — название модели
  llm_timeout   (str) — таймаут в секундах (default "300")
  max_retries   (str) — попыток при невалидном JSON (default "3")
Outputs: analysis (Object)
"""
import json
import re
import ssl
import urllib.request

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


# ── LLM ──────────────────────────────────────────────────────────────

def _call_llm(api_base, api_key, model, messages, timeout):
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text):
    """Пробует распарсить JSON; если не выходит — ищет первый {...} блок."""
    text = _strip_think(text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


# ── Валидация (как instructor) ────────────────────────────────────────

_SEVERITIES = {"critical", "high", "medium", "low", "info"}
_CONFIDENCES = {"low", "medium", "high"}


def _validate(data):
    """Проверяет структуру JSON. Возвращает список ошибок (пустой = ок)."""
    errors = []

    if not isinstance(data, dict):
        return ["root must be a JSON object"]

    # time_range
    tr = data.get("time_range")
    if not (isinstance(tr, list) and len(tr) == 2):
        errors.append("time_range must be a list of 2 ISO strings")

    # narrative
    if not isinstance(data.get("narrative"), str):
        errors.append("narrative must be a string")

    # events
    for i, ev in enumerate(data.get("events", [])):
        if not isinstance(ev.get("id"), str):
            errors.append(f"events[{i}].id must be a string")
        if ev.get("severity") not in _SEVERITIES:
            errors.append(f"events[{i}].severity must be one of {_SEVERITIES}")
        imp = ev.get("importance")
        if not isinstance(imp, (int, float)) or not (0.0 <= imp <= 1.0):
            errors.append(f"events[{i}].importance must be float 0.0-1.0")

    # evidence
    for i, ev in enumerate(data.get("evidence", [])):
        if not isinstance(ev.get("raw_line"), str):
            errors.append(f"evidence[{i}].raw_line must be a string")
        if ev.get("severity") not in _SEVERITIES:
            errors.append(f"evidence[{i}].severity must be one of {_SEVERITIES}")

    # hypotheses
    for i, h in enumerate(data.get("hypotheses", [])):
        if h.get("confidence") not in _CONFIDENCES:
            errors.append(f"hypotheses[{i}].confidence must be one of {_CONFIDENCES}")

    return errors



# ── Промпты ───────────────────────────────────────────────────────────

def _system(incident_info, period_start, period_end):
    return f"""You are a senior SRE analyzing a production log batch during an active incident.

=== Incident context ===
{incident_info}

Period under investigation: {period_start} → {period_end}

=== Language ===
Think and respond in English. Keep technical terms as-is (OOM, pod names, service names,
Kubernetes objects, error codes, metric names, CLI commands).

=== Task ===
Analyze the log batch provided by the user. Extract key events, evidence, and hypotheses.
Output a single JSON object — no prose before or after.

=== Output schema ===
{{
  "time_range": ["<ISO8601>", "<ISO8601>"],
  "narrative": "<3-5 sentences: what happened, specific timestamps and service names>",
  "events": [
    {{
      "id": "<evt-N>",
      "timestamp": "<ISO8601>",
      "source": "<service / pod / container>",
      "description": "<≤80 chars>",
      "severity": "critical|high|medium|low|info",
      "importance": <0.0-1.0>,
      "tags": []
    }}
  ],
  "evidence": [
    {{
      "id": "<ev-N>",
      "timestamp": "<ISO8601>",
      "source": "<service>",
      "raw_line": "<exact log line verbatim>",
      "severity": "critical|high|medium|low|info",
      "linked_event_id": "<evt-id or null>"
    }}
  ],
  "hypotheses": [
    {{
      "id": "<hyp-N>",
      "title": "<≤60 chars>",
      "description": "<failure scenario narrative>",
      "reasoning": "<specific evidence and timing>",
      "confidence": "low|medium|high",
      "supporting_event_ids": [],
      "contradicting_event_ids": [],
      "related_alert_ids": []
    }}
  ],
  "anomalies": [{{"description": "<what is unusual>", "related_event_ids": []}}],
  "preliminary_recommendations": ["<concrete action, ≤80 chars>"]
}}

=== Rules ===
- Skip healthy/routine lines (200 OK, heartbeat, etc.)
- evidence: max 10 verbatim log lines per batch
- hypotheses: only if concrete evidence exists
- Output ONLY the JSON object, nothing else
"""


def _user(item, period_start, period_end):
    return f"## Log batch  {period_start} → {period_end}\n\n```\n{item}\n```"


# ── Main ──────────────────────────────────────────────────────────────

def main(
    item: str,
    incident_info: str,
    period_start: str,
    period_end: str,
    llm_api_base: str = "http://localhost:8000",
    llm_api_key: str = "none",
    llm_model: str = "default",
    llm_timeout: str = "300",
    max_retries: str = "3",
) -> dict:
    system  = _system(incident_info, period_start, period_end)
    user    = _user(item, period_start, period_end)
    timeout = float(llm_timeout)
    retries = int(max_retries)

    # messages — накапливаем историю как instructor: при ошибке добавляем
    # assistant-ответ и user-сообщение с описанием ошибок
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    for _ in range(retries):
        # LLM-ошибки (коннект, авторизация, таймаут) — не глотаем, пробрасываем
        raw = _call_llm(llm_api_base, llm_api_key, llm_model, messages, timeout)

        try:
            analysis = _extract_json(raw)
        except Exception:
            # JSON не распарсился — просим переделать
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "Your response was not valid JSON. Return only a JSON object.",
            })
            continue

        errors = _validate(analysis)
        if not errors:
            return {"analysis": analysis}

        # Валидация не прошла — отправляем ошибки обратно в LLM (как instructor)
        error_text = "\n".join(f"- {e}" for e in errors)
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                f"Your response has validation errors, please fix them and return "
                f"the corrected JSON:\n{error_text}"
            ),
        })

    last_msg = messages[-1].get("content", "") if messages else ""
    raise RuntimeError(
        f"MAP failed after {retries} attempts. Last error: {last_msg[:300]}"
    )
