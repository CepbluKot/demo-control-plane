"""Compatibility exports for input ingestion services.

The concrete workflows live in domain-specific modules. Keep this facade so
API, workers, and tests can import from ``summary_backend.ingestion`` without
knowing the internal layout.
"""

from __future__ import annotations

from .ingestion_models import QueryIngestionResult, StagedUploadResult, UploadIngestionResult
from .query_ingestion import ClickHouseQueryIngestionService
from .staged_upload_ingestion import StagedUploadIngestionService
from .upload_ingestion import UploadedFileIngestionService

__all__ = [
    "ClickHouseQueryIngestionService",
    "QueryIngestionResult",
    "StagedUploadIngestionService",
    "StagedUploadResult",
    "UploadedFileIngestionService",
    "UploadIngestionResult",
]
