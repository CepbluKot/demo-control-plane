"""LLM and pipeline error classification."""

from __future__ import annotations


class SummaryBackendError(Exception):
    pass


class RetryableBackendError(SummaryBackendError):
    pass


class ContextOverflowError(RetryableBackendError):
    pass


class ProviderUnavailableError(RetryableBackendError):
    pass


class RateLimitError(RetryableBackendError):
    pass


class FatalBackendError(SummaryBackendError):
    pass


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "context" in text and ("length" in text or "too long" in text or "maximum" in text):
        return "context_too_long"
    if "rate limit" in text or "429" in text:
        return "rate_limit"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "503" in text or "502" in text or "500" in text or "connection" in text:
        return "provider_unavailable"
    if "json" in text or "validation" in text or "schema" in text:
        return "schema_or_json"
    return "unknown"
