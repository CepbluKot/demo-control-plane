from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.log-summarizer",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── ClickHouse source ─────────────────────────────────────────────
    ch_host: str = "localhost"
    ch_port: int = 8123
    ch_user: str = "default"
    ch_password: str = ""
    ch_database: str = "default"

    # ── ClickHouse results ────────────────────────────────────────────
    result_database: str = "default"
    save_to_ch: bool = False
    airflow_run_id: str = ""

    # ── LLM ──────────────────────────────────────────────────────────
    llm_api_base: str = "http://localhost:8000"
    llm_api_key: str = Field(default="sk-placeholder", alias="LLM_API_KEY")
    llm_model: str = "default-model"
    llm_context_tokens: int = 150000
    llm_tool_calling: bool = False

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str | None = None
