from __future__ import annotations

from urllib.parse import quote_plus

from clickhouse_sqlalchemy import get_declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from settings import settings


def _read_setting(name: str, default: str | None = None) -> str | None:
    raw = getattr(settings, name, None)
    if raw is None or raw == "":
        return default
    return str(raw)


db_username = _read_setting(
    "TGT_DATABASE_USERNAME",
    default="",
) or ""
db_password = _read_setting(
    "TGT_DATABASE_PASSWORD",
    default="",
) or ""
db_host = _read_setting(
    "TGT_DATABASE_HOST",
    default="localhost",
) or "localhost"
db_port = _read_setting(
    "TGT_DATABASE_PORT",
    default="8123",
) or "8123"
db_name = _read_setting(
    "METRICS_PREDICTIONS_DATABASE",
    default="default",
) or "default"

encoded_user = quote_plus(db_username)
encoded_pass = quote_plus(db_password)

if encoded_user:
    CLICKHOUSE_URL = f"clickhouse://{encoded_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}"
else:
    CLICKHOUSE_URL = f"clickhouse://{db_host}:{db_port}/{db_name}"

engine = create_engine(CLICKHOUSE_URL, echo=False)
Base = get_declarative_base()
Session = sessionmaker(bind=engine)
