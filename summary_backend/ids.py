"""Deterministic ids and content hashes."""

from __future__ import annotations

import hashlib
import uuid


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, length: int = 16) -> str:
    return sha256_text(text)[:length]


def make_node_id(
    *,
    job_id: str,
    node_type: str,
    level: int,
    index: int,
    input_hash: str,
) -> str:
    raw = "|".join([job_id, node_type, str(level), str(index), input_hash])
    return f"{node_type.lower()}_{short_hash(raw, 24)}"
