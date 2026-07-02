from __future__ import annotations

import os
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ.setdefault("SUMMARY_BACKEND_LOG_DIR", "/tmp/demo-control-plane-test-logs")
os.environ.setdefault("SUMMARY_BACKEND_AUDIT_DIR", "/tmp/demo-control-plane-test-audit")
os.environ.setdefault("SUMMARY_BACKEND_UPLOAD_STAGING_DIR", "/tmp/demo-control-plane-test-uploads")

import summary_backend.api as api_module
from summary_backend.config import get_settings
from summary_backend.dify_client import DifyWorkflowRunResult
from summary_backend.monitoring import MonitoringService, create_monitoring_service
from summary_backend.schemas import (
    CreateMonitoringProfileRequest,
    CreateMonitoringRunRequest,
    MonitoringRunStatus,
    MonitoringRunTrigger,
    MonitoringScheduleConfig,
)


class InMemoryMonitoringStore:
    def __init__(self) -> None:
        self.profile_snapshots: list[dict[str, Any]] = []
        self.schedule_snapshots: list[dict[str, Any]] = []
        self.run_snapshots: list[dict[str, Any]] = []
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _insert(self, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
        rows.append({"_seq": self._next_seq(), **payload})

    def insert_monitoring_profile_snapshot(self, **kwargs: Any) -> None:
        self._insert(self.profile_snapshots, kwargs)

    def _latest_by_key(self, rows: list[dict[str, Any]], key_name: str, key_value: str) -> dict[str, Any] | None:
        candidates = [row for row in rows if str(row.get(key_name) or "") == key_value]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row.get("updated_at"), row["_seq"]))

    def get_monitoring_profile_current(self, profile_id: str) -> dict[str, Any] | None:
        latest = self._latest_by_key(self.profile_snapshots, "profile_id", profile_id)
        if latest is None:
            return None
        created_at = min(
            row["created_at"]
            for row in self.profile_snapshots
            if str(row.get("profile_id") or "") == profile_id
        )
        return {**latest, "created_at": created_at}

    def list_monitoring_profiles(self, *, include_archived: bool = False, limit: int = 500) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.profile_snapshots:
            profile_id = str(row.get("profile_id") or "")
            current = latest.get(profile_id)
            if current is None or (row.get("updated_at"), row["_seq"]) > (current.get("updated_at"), current["_seq"]):
                latest[profile_id] = row
        rows = []
        for profile_id, row in latest.items():
            if not include_archived and bool(row.get("is_archived")):
                continue
            created_at = min(
                item["created_at"]
                for item in self.profile_snapshots
                if str(item.get("profile_id") or "") == profile_id
            )
            rows.append({**row, "created_at": created_at})
        rows.sort(key=lambda row: (row.get("updated_at"), row.get("profile_id")), reverse=True)
        return rows[:limit]

    def insert_monitoring_schedule_snapshot(self, **kwargs: Any) -> None:
        self._insert(self.schedule_snapshots, kwargs)

    def get_monitoring_schedule_current(self, profile_id: str) -> dict[str, Any] | None:
        latest = self._latest_by_key(self.schedule_snapshots, "profile_id", profile_id)
        if latest is None:
            return None
        created_at = min(
            row["created_at"]
            for row in self.schedule_snapshots
            if str(row.get("profile_id") or "") == profile_id
        )
        return {**latest, "created_at": created_at}

    def list_due_monitoring_schedules(self, *, now, limit: int = 50) -> list[dict[str, Any]]:
        latest: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in sorted(self.schedule_snapshots, key=lambda item: (item.get("updated_at"), item["_seq"]), reverse=True):
            profile_id = str(row.get("profile_id") or "")
            if profile_id in seen:
                continue
            seen.add(profile_id)
            if not bool(row.get("is_enabled")):
                continue
            next_run_at = row.get("next_run_at")
            if next_run_at is None or next_run_at > now:
                continue
            created_at = min(
                item["created_at"]
                for item in self.schedule_snapshots
                if str(item.get("profile_id") or "") == profile_id
            )
            latest.append({**row, "created_at": created_at})
        latest.sort(key=lambda row: (row.get("next_run_at"), row.get("profile_id")))
        return latest[:limit]

    def insert_monitoring_run_snapshot(self, **kwargs: Any) -> None:
        self._insert(self.run_snapshots, kwargs)

    def get_monitoring_run_current(self, run_id: str) -> dict[str, Any] | None:
        return self._latest_by_key(self.run_snapshots, "run_id", run_id)

    def list_monitoring_runs(
        self,
        *,
        profile_id: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.run_snapshots:
            run_id = str(row.get("run_id") or "")
            current = latest.get(run_id)
            if current is None or (row.get("updated_at"), row["_seq"]) > (current.get("updated_at"), current["_seq"]):
                latest[run_id] = row
        rows = list(latest.values())
        if profile_id:
            rows = [row for row in rows if str(row.get("profile_id") or "") == profile_id]
        if status:
            rows = [row for row in rows if str(row.get("status") or "") == status]
        rows.sort(key=lambda row: (row.get("updated_at"), row.get("run_id")), reverse=True)
        return rows[:limit]

    def count_monitoring_active_runs(self, profile_id: str) -> int:
        return sum(
            1
            for row in self.list_monitoring_runs(profile_id=profile_id, limit=10000)
            if str(row.get("status") or "") in {"CREATED", "RUNNING"}
        )

    def close(self) -> None:
        return None


class FakeWorkflowClient:
    def __init__(self, *, status: str = "succeeded", outputs: dict[str, Any] | None = None) -> None:
        self.status = status
        self.outputs = outputs or {"ok": True}
        self.calls: list[dict[str, Any]] = []

    def run_workflow(self, *, workflow_id: str, inputs: dict[str, Any], user: str, timeout_seconds: float | None = None):
        self.calls.append(
            {
                "workflow_id": workflow_id,
                "inputs": dict(inputs),
                "user": user,
                "timeout_seconds": timeout_seconds,
            }
        )
        return DifyWorkflowRunResult(
            status=self.status,
            workflow_run_id="wf_run_demo",
            task_id="wf_task_demo",
            outputs=dict(self.outputs),
            raw={"data": {"status": self.status, "outputs": dict(self.outputs)}},
        )


def wait_for_run(service: MonitoringService, run_id: str, timeout_seconds: float = 2.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run = service.get_run(run_id)
        if run and run.status in {MonitoringRunStatus.DONE, MonitoringRunStatus.FAILED}:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run did not finish in time: {run_id}")


class MonitoringControlPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryMonitoringStore()
        self.runner = FakeWorkflowClient()
        self.settings = replace(get_settings(), monitoring_scheduler_enabled=False)
        self.service = create_monitoring_service(store=self.store, settings=self.settings, workflow_client=self.runner)

    def test_create_profile_with_schedule_computes_next_run(self) -> None:
        profile = self.service.create_profile(
            CreateMonitoringProfileRequest(
                name="Payment Monitoring",
                service="payment-gateway",
                description="periodic monitoring",
                workflow_id="pa-control-plane-main-orchestrator",
                workflow_inputs={"summary_trigger_mode": "any_metric"},
                schedule=MonitoringScheduleConfig(enabled=True, cron="*/15 * * * *", timezone="UTC", max_active_runs=2),
            )
        )

        self.assertEqual(profile.service, "payment-gateway")
        self.assertEqual(profile.workflow_id, "pa-control-plane-main-orchestrator")
        self.assertIsNotNone(profile.next_run_at)
        self.assertEqual(profile.schedule.max_active_runs, 2)
        self.assertFalse(profile.is_archived)

    def test_scheduler_tick_launches_due_run_and_advances_schedule(self) -> None:
        profile = self.service.create_profile(
            CreateMonitoringProfileRequest(
                name="Infra Monitoring",
                service="infra-cluster",
                description="scheduler test",
                workflow_id="pa-control-plane-main-orchestrator",
                workflow_inputs={"metrics_sql_queries_json": "[]"},
                schedule=MonitoringScheduleConfig(enabled=True, cron="*/5 * * * *", timezone="UTC", max_active_runs=1),
            )
        )
        schedule_row = self.store.get_monitoring_schedule_current(profile.profile_id)
        assert schedule_row is not None
        due_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.store.insert_monitoring_schedule_snapshot(
            profile_id=profile.profile_id,
            cron=schedule_row["cron"],
            timezone=schedule_row["timezone"],
            is_enabled=True,
            max_active_runs=1,
            next_run_at=due_at,
            last_run_at=schedule_row.get("last_run_at"),
            created_at=schedule_row["created_at"],
            updated_at=datetime.now(timezone.utc),
        )

        tick = self.service.run_due_schedules(now=datetime.now(timezone.utc))

        self.assertEqual(tick.launched, 1)
        self.assertEqual(tick.skipped, 0)
        run_id = tick.items[0].run_id
        run = wait_for_run(self.service, run_id)
        self.assertEqual(run.status, MonitoringRunStatus.DONE)
        self.assertEqual(run.trigger_type, MonitoringRunTrigger.SCHEDULED)
        self.assertEqual(self.runner.calls[0]["workflow_id"], "pa-control-plane-main-orchestrator")
        schedule_after = self.store.get_monitoring_schedule_current(profile.profile_id)
        self.assertIsNotNone(schedule_after["last_run_at"])
        self.assertGreater(schedule_after["next_run_at"], due_at)

    def test_scheduler_skips_when_max_active_runs_reached(self) -> None:
        profile = self.service.create_profile(
            CreateMonitoringProfileRequest(
                name="Logs Monitoring",
                service="summary-service",
                description="skip test",
                workflow_id="pa-general-summary-generator",
                workflow_inputs={"raw_input": "[]"},
                schedule=MonitoringScheduleConfig(enabled=True, cron="*/5 * * * *", timezone="UTC", max_active_runs=1),
            )
        )
        self.service.create_run(
            profile.profile_id,
            CreateMonitoringRunRequest(workflow_inputs_override={}),
            trigger_type=MonitoringRunTrigger.MANUAL,
            auto_start=False,
        )
        schedule_row = self.store.get_monitoring_schedule_current(profile.profile_id)
        assert schedule_row is not None
        self.store.insert_monitoring_schedule_snapshot(
            profile_id=profile.profile_id,
            cron=schedule_row["cron"],
            timezone=schedule_row["timezone"],
            is_enabled=True,
            max_active_runs=1,
            next_run_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            last_run_at=schedule_row.get("last_run_at"),
            created_at=schedule_row["created_at"],
            updated_at=datetime.now(timezone.utc),
        )

        tick = self.service.run_due_schedules(now=datetime.now(timezone.utc))

        self.assertEqual(tick.launched, 0)
        self.assertEqual(tick.skipped, 1)
        self.assertEqual(tick.items[0].detail, "max_active_runs_reached")

    def test_api_create_profile_run_and_archive(self) -> None:
        test_service = self.service
        with patch.object(api_module, "monitoring_service", test_service):
            client = TestClient(api_module.app)

            create_response = client.post(
                "/monitoring-profiles",
                json={
                    "name": "API Monitoring",
                    "service": "checkout-api",
                    "description": "api test",
                    "workflow_id": "pa-control-plane-main-orchestrator",
                    "workflow_inputs": {"metrics_sql_queries_json": "[]"},
                    "schedule": {
                        "enabled": True,
                        "cron": "0 * * * *",
                        "timezone": "UTC",
                        "max_active_runs": 1,
                    },
                },
            )
            self.assertEqual(create_response.status_code, 200)
            profile_id = create_response.json()["profile"]["profile_id"]

            run_response = client.post(
                f"/monitoring-profiles/{profile_id}/run",
                json={"workflow_inputs_override": {"summary_trigger_mode": "any_metric"}},
            )
            self.assertEqual(run_response.status_code, 200)
            run_id = run_response.json()["run"]["run_id"]
            run = wait_for_run(test_service, run_id)
            self.assertEqual(run.status, MonitoringRunStatus.DONE)

            list_response = client.get("/monitoring-runs", params={"profile_id": profile_id})
            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(len(list_response.json()), 1)

            archive_response = client.post(f"/monitoring-profiles/{profile_id}/archive")
            self.assertEqual(archive_response.status_code, 200)
            self.assertTrue(archive_response.json()["profile"]["is_archived"])

    def test_api_rejects_invalid_schedule(self) -> None:
        with patch.object(api_module, "monitoring_service", self.service):
            client = TestClient(api_module.app)
            response = client.post(
                "/monitoring-profiles",
                json={
                    "name": "Broken cron",
                    "service": "bad-service",
                    "description": "",
                    "workflow_id": "pa-control-plane-main-orchestrator",
                    "workflow_inputs": {},
                    "schedule": {
                        "enabled": True,
                        "cron": "not a cron",
                        "timezone": "UTC",
                        "max_active_runs": 1,
                    },
                },
            )
            self.assertEqual(response.status_code, 422)
            self.assertIn("invalid cron expression", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
