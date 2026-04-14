from __future__ import annotations

"""
Quick benchmark: compare LLM speed in tokens/sec on the same prompt.

Run:
  ./venv/bin/python debug_llm_tokens_per_sec.py
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import statistics
import time
from typing import Any, Dict, List, Optional

import requests

from settings import settings


@dataclass
class BenchmarkConfig:
    # Задай тут напрямую, чтобы не использовать .env
    api_base: str = ""  # Пример: "https://phoenix.scm-test.int.gazprombank.ru/api/v1"
    api_key: str = ""  # Пример: "sk-..."
    models: List[str] = field(
        default_factory=lambda: [
            "PNX.QWEN3 235b a22b instruct",
            # Add more models here for comparison:
            # "PNX.DeepSeek V3.2",
            # "another-model-id",
        ]
    )
    system_prompt: str = "You are a concise and accurate assistant."
    user_prompt: str = (
        "Explain the same technical topic in detail: "
        "how map-reduce summarization pipelines fail and recover. "
        "Give a structured answer with 8-10 sections."
    )
    max_tokens: int = 1200
    temperature: float = 0.1
    runs_per_model: int = 3
    timeout_sec: float = 1200.0
    sleep_between_runs_sec: float = 1.0
    approx_chars_per_token: float = 3.5
    output_dir: str = "artifacts/llm_speed_benchmark"


RUN_CONFIG = BenchmarkConfig(
    api_base="",
    api_key="",
    models=[
        "PNX.QWEN3 235b a22b instruct",
    ],
)


def _build_chat_completions_url(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_text_from_response(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text is None:
                            text = item.get("content")
                        if text is not None:
                            parts.append(str(text))
                    elif item is not None:
                        parts.append(str(item))
                return "".join(parts)
        text_fallback = first.get("text")
        if text_fallback is not None:
            return str(text_fallback)
    output_text = data.get("output_text")
    if output_text is not None:
        return str(output_text)
    return ""


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def _bench_one_call(
    *,
    url: str,
    api_key: str,
    model: str,
    cfg: BenchmarkConfig,
    run_no: int,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": cfg.user_prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }

    started = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_sec)
    except Exception as exc:  # noqa: BLE001
        latency_sec = max(time.perf_counter() - started, 0.0)
        return {
            "model": model,
            "run_no": run_no,
            "ok": False,
            "status_code": None,
            "latency_sec": latency_sec,
            "error": f"{type(exc).__name__}: {exc}",
        }

    latency_sec = max(time.perf_counter() - started, 0.0)
    status_code = int(getattr(response, "status_code", 0) or 0)
    body_text = str(getattr(response, "text", "") or "")

    if status_code >= 400:
        return {
            "model": model,
            "run_no": run_no,
            "ok": False,
            "status_code": status_code,
            "latency_sec": latency_sec,
            "error": body_text[:1500],
        }

    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model,
            "run_no": run_no,
            "ok": False,
            "status_code": status_code,
            "latency_sec": latency_sec,
            "error": f"invalid json: {type(exc).__name__}: {exc}",
            "response_preview": body_text[:1500],
        }

    usage = data.get("usage") if isinstance(data, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    response_text = _extract_text_from_response(data if isinstance(data, dict) else {})
    token_source = "reported"
    if completion_tokens is None:
        token_source = "estimated_chars_div"
        completion_tokens = max(int(round(len(response_text) / max(cfg.approx_chars_per_token, 0.1))), 0)
    if prompt_tokens is None and total_tokens is None:
        prompt_tokens = None
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    completion_tps = None
    total_tps = None
    if latency_sec > 0 and completion_tokens is not None:
        completion_tps = float(completion_tokens) / latency_sec
    if latency_sec > 0 and total_tokens is not None:
        total_tps = float(total_tokens) / latency_sec

    return {
        "model": model,
        "run_no": run_no,
        "ok": True,
        "status_code": status_code,
        "latency_sec": latency_sec,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tps": completion_tps,
        "total_tps": total_tps,
        "token_source": token_source,
        "response_chars": len(response_text),
    }


def main() -> int:
    cfg = RUN_CONFIG
    api_base = str(cfg.api_base or "").strip() or str(getattr(settings, "OPENAI_API_BASE_DB", "") or "").strip()
    api_key = str(cfg.api_key or "").strip() or str(getattr(settings, "OPENAI_API_KEY_DB", "") or "").strip()
    if not api_base or not api_key:
        raise RuntimeError("Set RUN_CONFIG.api_base and RUN_CONFIG.api_key in code.")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = _build_chat_completions_url(api_base)

    print(f"Benchmark start: {url}")
    print(f"Models: {', '.join(cfg.models)}")
    print(f"Runs per model: {cfg.runs_per_model}")

    rows: List[Dict[str, Any]] = []
    for model in cfg.models:
        model_name = str(model or "").strip()
        if not model_name:
            continue
        for run_no in range(1, max(int(cfg.runs_per_model), 1) + 1):
            row = _bench_one_call(url=url, api_key=api_key, model=model_name, cfg=cfg, run_no=run_no)
            rows.append(row)
            if row.get("ok"):
                print(
                    f"[OK] model={model_name} run={run_no} "
                    f"lat={row.get('latency_sec'):.2f}s "
                    f"completion_tps={row.get('completion_tps'):.2f} "
                    f"total_tps={row.get('total_tps'):.2f} "
                    f"tokens={row.get('completion_tokens')}/{row.get('total_tokens')}"
                )
            else:
                print(
                    f"[ERR] model={model_name} run={run_no} "
                    f"lat={row.get('latency_sec'):.2f}s status={row.get('status_code')} "
                    f"error={str(row.get('error') or '')[:180]}"
                )
            if cfg.sleep_between_runs_sec > 0:
                time.sleep(float(cfg.sleep_between_runs_sec))

    summary: Dict[str, Any] = {"models": []}
    for model in cfg.models:
        model_rows = [r for r in rows if r.get("model") == model and r.get("ok")]
        lat_values = [float(r["latency_sec"]) for r in model_rows if r.get("latency_sec") is not None]
        ctps_values = [float(r["completion_tps"]) for r in model_rows if r.get("completion_tps") is not None]
        ttps_values = [float(r["total_tps"]) for r in model_rows if r.get("total_tps") is not None]

        summary["models"].append(
            {
                "model": model,
                "runs_ok": len(model_rows),
                "runs_total": len([r for r in rows if r.get("model") == model]),
                "latency_sec_mean": statistics.mean(lat_values) if lat_values else None,
                "latency_sec_median": statistics.median(lat_values) if lat_values else None,
                "latency_sec_p95": _p95(lat_values),
                "completion_tps_mean": statistics.mean(ctps_values) if ctps_values else None,
                "completion_tps_median": statistics.median(ctps_values) if ctps_values else None,
                "completion_tps_p95": _p95(ctps_values),
                "total_tps_mean": statistics.mean(ttps_values) if ttps_values else None,
                "total_tps_median": statistics.median(ttps_values) if ttps_values else None,
                "total_tps_p95": _p95(ttps_values),
            }
        )

    raw_path = out_dir / "raw_runs.json"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"

    raw_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg_dump = asdict(cfg)
    if cfg_dump.get("api_key"):
        cfg_dump["api_key"] = "***"
    config_path.write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary:")
    for item in summary["models"]:
        print(
            f"- {item['model']}: "
            f"ok={item['runs_ok']}/{item['runs_total']}, "
            f"completion_tps_mean={item['completion_tps_mean']}, "
            f"total_tps_mean={item['total_tps_mean']}, "
            f"lat_mean={item['latency_sec_mean']}"
        )

    print(f"\nSaved:\n- {raw_path}\n- {summary_path}\n- {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
