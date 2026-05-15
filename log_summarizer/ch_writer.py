"""Сохранение отчёта в ClickHouse через clickhouse_sqlalchemy."""

from __future__ import annotations

import logging
from functools import lru_cache

from clickhouse_sqlalchemy import engines as ch_engines
from clickhouse_sqlalchemy import get_declarative_base
from clickhouse_sqlalchemy import types as ch_types
from sqlalchemy import Column, String, create_engine, text
from sqlalchemy.orm import Session

logger = logging.getLogger("ch_writer")


@lru_cache(maxsize=None)
def _model(database: str):
    Base = get_declarative_base()

    class LogSummarizerRun(Base):
        __tablename__ = "log_summarizer_runs"
        __table_args__ = (
            ch_engines.MergeTree(order_by=("created_at",)),
            {"schema": database},
        )

        run_id           = Column(String, primary_key=True)
        created_at       = Column(ch_types.DateTime64(3), server_default=text("now64()"))
        incident_context = Column(String)
        incident_start   = Column(String)
        incident_end     = Column(String)
        report           = Column(String)
        logs_sql         = Column(String)

    return Base, LogSummarizerRun


def save_report(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    result_database: str,
    run_id: str,
    incident_context: str,
    incident_start: str,
    incident_end: str,
    report: str,
    logs_sql: str = "",
) -> None:
    url = f"clickhouse+http://{user}:{password}@{host}:{port}/{result_database}"
    engine = create_engine(url)

    Base, LogSummarizerRun = _model(result_database)
    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        session.add(LogSummarizerRun(
            run_id=run_id,
            incident_context=incident_context,
            incident_start=incident_start,
            incident_end=incident_end,
            report=report,
            logs_sql=logs_sql,
        ))
        session.commit()

    logger.info("Report saved to ClickHouse: run_id=%s", run_id)
