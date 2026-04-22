import json
import urllib.error
import urllib.request


class LLMError(Exception):
    """LLM недоступен или вернул ошибку."""


def call_llm(
    api_base: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout: float = 2400,
    temperature: float = 0.2,
) -> str:
    """Вызывает OpenAI-совместимый LLM API. Возвращает строку content."""
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        raise LLMError(str(exc)) from exc
