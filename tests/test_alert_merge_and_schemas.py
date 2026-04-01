import unittest

from alert_merge import merge_alert_refs
from schemas import AlertRef, Context, DataQuality, IncidentSummary, TimelineEvent


def _make_summary(
    *,
    batch_id: str,
    event_id: str,
    alert_status: str,
    explanation: str,
) -> IncidentSummary:
    return IncidentSummary(
        context=Context(
            batch_id=batch_id,
            time_range_start="2026-03-25T10:00:00+00:00",
            time_range_end="2026-03-25T10:05:00+00:00",
            total_log_entries=1,
            source_query=["SELECT * FROM logs"],
            source_services=["svc-a"],
        ),
        timeline=[
            TimelineEvent(
                id=event_id,
                timestamp="2026-03-25T10:00:00+00:00",
                source="svc-a",
                description="event",
                severity="low",
                importance=0.8,
                evidence_type="HYPOTHESIS",
                tags=["test"],
            )
        ],
        alert_refs=[
            AlertRef(
                alert_id="A1",
                status=alert_status,
                related_events=[event_id],
                explanation=explanation,
            )
        ],
        data_quality=DataQuality(is_empty=False, noise_ratio=0.1, notes=""),
    )


class TestAlertMergeAndSchemas(unittest.TestCase):
    def test_merge_alert_refs_uses_status_priority_and_union_events(self) -> None:
        s1 = _make_summary(
            batch_id="batch-1",
            event_id="e1",
            alert_status="PARTIALLY",
            explanation="from batch 1",
        )
        s2 = _make_summary(
            batch_id="batch-2",
            event_id="e2",
            alert_status="EXPLAINED",
            explanation="from batch 2",
        )
        merged = merge_alert_refs([s1, s2])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].alert_id, "A1")
        self.assertEqual(merged[0].status, "EXPLAINED")
        self.assertEqual(merged[0].related_events, ["e1", "e2"])
        self.assertIn("from batch 1", merged[0].explanation)
        self.assertIn("from batch 2", merged[0].explanation)
        self.assertIn("|||", merged[0].explanation)

    def test_merge_alert_refs_applies_event_id_remap(self) -> None:
        s1 = _make_summary(
            batch_id="batch-1",
            event_id="old-eid",
            alert_status="NOT_EXPLAINED",
            explanation="explanation",
        )
        merged = merge_alert_refs([s1], id_remap={"old-eid": "new-eid"})
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].related_events, ["new-eid"])

    def test_schema_fact_event_requires_evidence_quote(self) -> None:
        with self.assertRaises(ValueError):
            TimelineEvent(
                id="evt-1",
                timestamp="2026-03-25T10:00:00+00:00",
                source="svc-a",
                description="fact event",
                severity="high",
                importance=0.9,
                evidence_type="FACT",
                tags=["error"],
            )

    def test_schema_rejects_unknown_alert_related_event(self) -> None:
        with self.assertRaises(ValueError):
            IncidentSummary(
                context=Context(
                    batch_id="batch-1",
                    time_range_start="2026-03-25T10:00:00+00:00",
                    time_range_end="2026-03-25T10:05:00+00:00",
                    total_log_entries=1,
                    source_query=[],
                    source_services=[],
                ),
                timeline=[
                    TimelineEvent(
                        id="evt-1",
                        timestamp="2026-03-25T10:00:00+00:00",
                        source="svc-a",
                        description="event",
                        severity="low",
                        importance=0.5,
                        evidence_type="HYPOTHESIS",
                        tags=["test"],
                    )
                ],
                alert_refs=[
                    AlertRef(
                        alert_id="A1",
                        status="PARTIALLY",
                        related_events=["evt-unknown"],
                        explanation="bad ref",
                    )
                ],
                data_quality=DataQuality(is_empty=False, noise_ratio=0.1, notes=""),
            )


if __name__ == "__main__":
    unittest.main()
