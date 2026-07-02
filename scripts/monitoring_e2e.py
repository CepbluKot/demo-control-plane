#!/usr/bin/env python3
"""Live e2e check for monitoring profiles executed via Dify."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import error, request

import clickhouse_connect


def http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=120) as response:
        text = response.read().decode("utf-8")
    return json.loads(text) if text else {}


def wait_for_run(base_url: str, run_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = http_json("GET", f"{base_url.rstrip('/')}/monitoring-runs/{run_id}")
        status = str(payload.get("status") or "").upper()
        if status in {"DONE", "FAILED", "SKIPPED"}:
            return payload
        time.sleep(2.0)
    raise TimeoutError(f"Run {run_id} did not finish within {timeout_seconds} seconds.")


def force_schedule_due(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    database: str,
    profile_id: str,
    cron: str,
    timezone_name: str,
    max_active_runs: int,
) -> None:
    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
    )
    now = datetime.now(timezone.utc)
    due_at = now - timedelta(seconds=30)
    client.insert(
        f"{database}.monitoring_schedules",
        [[profile_id, cron, timezone_name, 1, max_active_runs, due_at, now, now, now]],
        column_names=[
            "profile_id",
            "cron",
            "timezone",
            "is_enabled",
            "max_active_runs",
            "next_run_at",
            "last_run_at",
            "created_at",
            "updated_at",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live monitoring-profile e2e against demo-control-plane.")
    parser.add_argument("--backend-url", default="http://localhost:28088", help="Summary backend base URL.")
    parser.add_argument("--clickhouse-host", default="localhost", help="ClickHouse host for backdating schedule.")
    parser.add_argument("--clickhouse-port", type=int, default=28123, help="ClickHouse HTTP port.")
    parser.add_argument("--clickhouse-user", default="default")
    parser.add_argument("--clickhouse-password", default="")
    parser.add_argument("--clickhouse-database", default="summary_test")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()

    backend_url = args.backend_url.rstrip("/")
    health = http_json("GET", f"{backend_url}/health")
    print("health:", json.dumps(health, ensure_ascii=False))

    workflow_inputs = {
        "title": "Monitoring E2E Report",
        "input_text": json.dumps(
            [
                {
                    "timestamp": "2026-07-02T00:00:00Z",
                    "service": "payment-gateway",
                    "message": "payment-gateway latency spike around deployment window",
                },
                {
                    "timestamp": "2026-07-02T00:01:00Z",
                    "service": "payment-gateway",
                    "message": "payment-gateway recovered after rollback",
                },
            ],
            ensure_ascii=False,
        ),
        "metadata_json": json.dumps({"service": "payment-gateway", "source": "monitoring_e2e"}, ensure_ascii=False),
        "poll_interval_seconds": 5,
        "timeout_seconds": 180,
    }
    cron = "*/5 * * * *"
    timezone_name = "UTC"
    max_active_runs = 1

    profile_payload = {
        "name": "Monitoring E2E Summary Profile",
        "service": "payment-gateway",
        "description": "Live e2e profile that runs Dify summary workflow and scheduler tick.",
        "workflow_id": "pa-general-summary-generator",
        "workflow_inputs": workflow_inputs,
        "schedule": {
            "enabled": True,
            "cron": cron,
            "timezone": timezone_name,
            "max_active_runs": max_active_runs,
        },
        "metadata": {"scenario": "monitoring_e2e"},
    }

    created = http_json("POST", f"{backend_url}/monitoring-profiles", profile_payload)
    profile = created["profile"]
    profile_id = profile["profile_id"]
    print("profile_id:", profile_id)

    manual = http_json("POST", f"{backend_url}/monitoring-profiles/{profile_id}/run", {"workflow_inputs_override": {}})
    manual_run_id = manual["run"]["run_id"]
    print("manual_run_id:", manual_run_id)
    manual_result = wait_for_run(backend_url, manual_run_id, args.timeout_seconds)
    print("manual_status:", manual_result["status"])
    if manual_result["status"] != "DONE":
        print(json.dumps(manual_result, ensure_ascii=False, indent=2))
        return 1

    force_schedule_due(
        host=args.clickhouse_host,
        port=args.clickhouse_port,
        username=args.clickhouse_user,
        password=args.clickhouse_password,
        database=args.clickhouse_database,
        profile_id=profile_id,
        cron=cron,
        timezone_name=timezone_name,
        max_active_runs=max_active_runs,
    )
    tick = http_json("POST", f"{backend_url}/monitoring-scheduler/tick?limit=50", {})
    print("tick:", json.dumps(tick, ensure_ascii=False))
    launched_items = [item for item in tick.get("items", []) if item.get("profile_id") == profile_id and item.get("action") == "launched"]
    if not launched_items:
        print("scheduler did not launch a run for the test profile", file=sys.stderr)
        return 1

    scheduled_run_id = launched_items[0]["run_id"]
    print("scheduled_run_id:", scheduled_run_id)
    scheduled_result = wait_for_run(backend_url, scheduled_run_id, args.timeout_seconds)
    print("scheduled_status:", scheduled_result["status"])
    if scheduled_result["status"] != "DONE":
        print(json.dumps(scheduled_result, ensure_ascii=False, indent=2))
        return 1

    runs = http_json("GET", f"{backend_url}/monitoring-runs?profile_id={profile_id}")
    print("runs_total:", len(runs))
    print("OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {body}", file=sys.stderr)
        raise SystemExit(1)
