"""File audit for LLM calls and backend debugging."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .ids import sha256_text
from .text import estimate_tokens


class AuditWriter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.audit_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def write_llm_call(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        request_json: dict[str, Any],
        response_json: dict[str, Any] | None,
        content: str | None,
        error: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        safe_node_id = node_id or "job"
        call_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:10]}"
        call_dir = self.root / job_id / safe_node_id / f"{stage.lower()}_{call_id}"
        call_dir.mkdir(parents=True, exist_ok=True)

        sys_tokens = estimate_tokens(system)
        user_tokens = estimate_tokens(user)
        content_tokens = estimate_tokens(content) if content else 0
        header = (
            f"# estimated_tokens: system={sys_tokens:,} user={user_tokens:,}"
            f" response={content_tokens:,} total={sys_tokens + user_tokens + content_tokens:,}\n"
            f"# system_sha256={sha256_text(system)}\n"
            f"# user_sha256={sha256_text(user)}\n\n"
        )

        paths: dict[str, str] = {}
        paths["system"] = self._write(call_dir / "system.txt", header + system)
        paths["user"] = self._write(call_dir / "user.txt", header + user)
        paths["request"] = self._write_json(call_dir / "request.json", request_json)

        if response_json is not None:
            paths["response"] = self._write_json(call_dir / "response.json", response_json)
        if content is not None:
            paths["content"] = self._write(call_dir / "content.txt", header + content)
        if error is not None:
            paths["error"] = self._write(call_dir / "error.txt", error)

        meta = dict(metadata or {})
        meta.update(
            {
                "job_id": job_id,
                "node_id": node_id,
                "stage": stage,
                "system_tokens_estimated": sys_tokens,
                "user_tokens_estimated": user_tokens,
                "response_tokens_estimated": content_tokens,
            }
        )
        paths["metadata"] = self._write_json(call_dir / "metadata.json", meta)
        return paths

    @staticmethod
    def _write(path: Path, content: str) -> str:
        path.write_text(content, encoding="utf-8")
        return str(path)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> str:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
