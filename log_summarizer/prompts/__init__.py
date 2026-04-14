"""Промпты для MAP / REDUCE / Report фаз."""

from log_summarizer.prompts.map_system import MAP_SYSTEM_TEMPLATE
from log_summarizer.prompts.map_user import format_map_user_prompt
from log_summarizer.prompts.reduce_merge import REDUCE_MERGE_SYSTEM_TEMPLATE
from log_summarizer.prompts.reduce_compress import REDUCE_COMPRESS_SYSTEM_TEMPLATE
from log_summarizer.prompts.report import (
    build_report_system_prompt,
    format_report_user_prompt,
)

__all__ = [
    "MAP_SYSTEM_TEMPLATE",
    "format_map_user_prompt",
    "REDUCE_MERGE_SYSTEM_TEMPLATE",
    "REDUCE_COMPRESS_SYSTEM_TEMPLATE",
    "build_report_system_prompt",
    "format_report_user_prompt",
]
