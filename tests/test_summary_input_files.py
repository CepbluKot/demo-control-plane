from __future__ import annotations

import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from summary_backend.config import get_settings
from summary_backend.ingestion import StagedUploadIngestionService, UploadedFileIngestionService
from summary_backend.input_parsers import CsvLogParser, JsonLogParser, MarkdownTableLogParser, ParserRegistry, PlainTextLogParser
from summary_backend.input_segments import RowBudgetInputSegmenter
from summary_backend.schemas import ArtifactType, JobStatus
from tests.test_summary_backend import InMemorySummaryStore, ManualQueue


class SummaryInputFileParserTests(unittest.TestCase):
    def test_csv_parser_reads_dbeaver_multiline_raw_line(self) -> None:
        csv_text = (
            '"timestamp","end_time","container_name","pod_name","raw_line"\n'
            '"2026-06-01 00:00:10.636 +0300","2026-06-01 00:00:10.636 +0300",'
            '"spark-kubernetes-driver","driver-pod","first line\nsecond line"\n'
        )

        records = list(CsvLogParser().parse_text_stream(io.StringIO(csv_text)))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].container_name, "spark-kubernetes-driver")
        self.assertEqual(records[0].pod_name, "driver-pod")
        self.assertIn("second line", records[0].raw_line)

    def test_markdown_parser_reads_table_export(self) -> None:
        markdown = """|timestamp|end_time|namespace|container_name|raw_line|
|---------|--------|---------|--------------|--------|
|2026-03-18 16:00:00.157|2026-03-18 16:00:00.158|kube-system|cilium-agent|level=warning msg="connection lost"|
"""

        records = list(MarkdownTableLogParser().parse_text_stream(io.StringIO(markdown)))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].namespace, "kube-system")
        self.assertEqual(records[0].container_name, "cilium-agent")
        self.assertIn("connection lost", records[0].render())

    def test_markdown_parser_preserves_backslashes_and_escaped_pipes(self) -> None:
        markdown = """|timestamp|raw_line|
|---------|--------|
|2026-03-18 16:00:00.157|path=C:\\tmp\\file msg=a\\|b|
"""

        records = list(MarkdownTableLogParser().parse_text_stream(io.StringIO(markdown)))

        self.assertIn(r"C:\tmp\file", records[0].raw_line)
        self.assertIn("a|b", records[0].raw_line)

    def test_json_parser_reads_dbeaver_query_object(self) -> None:
        payload = {
            "SELECT ...": [
                {
                    "timestamp": "2026-03-18T13:00:00.157Z",
                    "end_time": "2026-03-18T13:00:00.158Z",
                    "namespace": "kube-system",
                    "container_name": "cilium-agent",
                    "raw_line": "level=error msg=k8sError",
                }
            ]
        }

        records = list(JsonLogParser().parse_text_stream(io.StringIO(json.dumps(payload))))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].timestamp, "2026-03-18T13:00:00.157Z")
        self.assertIn("k8sError", records[0].raw_line)

    def test_parser_registry_detects_format(self) -> None:
        registry = ParserRegistry.default()

        self.assertEqual(registry.detect_format(filename="logs.csv", content_type="", requested_format="auto"), "csv")
        self.assertEqual(registry.detect_format(filename="logs.md", content_type="", requested_format="auto"), "markdown")
        self.assertEqual(registry.detect_format(filename="logs.json", content_type="", requested_format="auto"), "json")
        self.assertEqual(registry.detect_format(filename="logs.txt", content_type="text/plain", requested_format="auto"), "plain_text")
        self.assertEqual(registry.detect_format(filename="logs.txt", content_type="application/json", requested_format="auto"), "json")

    def test_plain_text_parser_reads_non_empty_lines(self) -> None:
        records = list(PlainTextLogParser().parse_text_stream(io.StringIO("first\n\n  second\n")))

        self.assertEqual([record.raw_line for record in records], ["first", "  second"])

    def test_segmenter_preserves_rows_and_source_metadata(self) -> None:
        records = list(
            CsvLogParser().parse_text_stream(
                io.StringIO(
                    "timestamp,namespace,container_name,raw_line\n"
                    "2026-01-01 00:00:00,ns,api,first\n"
                    "2026-01-01 00:00:01,ns,api,second\n"
                )
            )
        )

        segments = list(
            RowBudgetInputSegmenter().build_segments(
                records,
                source_type="upload",
                source_format="csv",
                target_estimated_tokens=256,
            )
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].rows_count, 2)
        self.assertEqual(segments[0].source_type, "upload")
        self.assertEqual(segments[0].source_format, "csv")
        self.assertIn("first", segments[0].content)
        self.assertIn("second", segments[0].content)

    def test_unknown_columns_are_rendered_as_attrs(self) -> None:
        records = list(
            CsvLogParser().parse_text_stream(
                io.StringIO("timestamp,service,severity,raw_line\n2026-01-01,payments,error,boom\n")
            )
        )

        rendered = records[0].render()

        self.assertIn("service=payments", rendered)
        self.assertIn("severity=error", rendered)

    def test_upload_ingestion_creates_job_manifest_and_input_segments(self) -> None:
        store = InMemorySummaryStore()
        queue = ManualQueue()
        settings = replace(get_settings(), chunk_target_estimated_tokens=256)
        service = UploadedFileIngestionService(store=store, queue=queue, settings=settings)
        content = (
            b"timestamp,namespace,container_name,raw_line\n"
            b"2026-01-01 00:00:00,ns,api,failed to connect\n"
            b"2026-01-01 00:00:01,ns,api,retry succeeded\n"
        )

        result = service.create_job_from_upload(
            file=io.BytesIO(content),
            filename="logs.csv",
            content_type="text/csv",
            title="upload",
            metadata={"case": "ingestion"},
            auto_start=True,
        )

        self.assertEqual(store.get_job_current(result.job_id)["job_status"], JobStatus.CREATED)
        self.assertEqual(result.rows_count, 2)
        self.assertEqual(store.count_input_segments(result.job_id), 1)
        self.assertEqual(queue.items[0], ("advance", result.job_id, None))
        input_artifact = store.latest_artifact(job_id=result.job_id, artifact_type=ArtifactType.INPUT)
        self.assertIsNotNone(input_artifact)
        self.assertIn("uploaded file manifest", input_artifact["content"])
        self.assertIn("failed to connect", store.list_input_segments(job_id=result.job_id, include_content=True)[0]["content"])

    def test_staged_upload_ingestion_runs_in_background_step(self) -> None:
        store = InMemorySummaryStore()
        queue = ManualQueue()
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = replace(
                get_settings(),
                chunk_target_estimated_tokens=256,
                upload_staging_dir=Path(tmp_dir),
            )
            service = StagedUploadIngestionService(store=store, queue=queue, settings=settings)
            content = (
                b"timestamp,namespace,container_name,raw_line\n"
                b"2026-01-01 00:00:00,ns,api,failed to connect\n"
                b"2026-01-01 00:00:01,ns,api,retry succeeded\n"
            )

            result = service.create_staged_upload_job(
                file=io.BytesIO(content),
                filename="logs.csv",
                content_type="text/csv",
                title="staged upload",
                metadata={"case": "staged"},
                auto_start=True,
            )

            self.assertEqual(store.get_job_current(result.job_id)["job_status"], JobStatus.INGESTING)
            self.assertEqual(queue.items[0], ("ingest", result.job_id, None))
            self.assertTrue((Path(tmp_dir) / result.job_id / "logs.csv").exists())

            service.ingest_staged_upload(result.job_id)

            self.assertEqual(store.get_job_current(result.job_id)["job_status"], JobStatus.INPUT_READY)
            self.assertEqual(store.count_input_segments(result.job_id), 1)
            self.assertIn("failed to connect", store.list_input_segments(job_id=result.job_id, include_content=True)[0]["content"])
            input_artifact = store.latest_artifact(job_id=result.job_id, artifact_type=ArtifactType.INPUT)
            self.assertIsNotNone(input_artifact)
            self.assertIn("staged upload manifest", input_artifact["content"])
            self.assertIn(("advance", result.job_id, None), queue.items)

    def test_existing_staged_upload_can_create_new_job(self) -> None:
        store = InMemorySummaryStore()
        queue = ManualQueue()
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = replace(
                get_settings(),
                chunk_target_estimated_tokens=256,
                upload_staging_dir=Path(tmp_dir),
            )
            service = StagedUploadIngestionService(store=store, queue=queue, settings=settings)
            content = (
                b"timestamp,namespace,container_name,raw_line\n"
                b"2026-01-01 00:00:00,ns,api,first failure\n"
                b"2026-01-01 00:00:01,ns,api,second failure\n"
            )

            original = service.create_staged_upload_job(
                file=io.BytesIO(content),
                filename="reusable.csv",
                content_type="text/csv",
                title="original upload",
                metadata={"case": "reuse-source"},
                auto_start=False,
            )

            uploads = store.list_staged_uploads()
            self.assertEqual(uploads[0]["upload_id"], original.job_id)
            self.assertEqual(uploads[0]["filename"], "reusable.csv")
            self.assertTrue(uploads[0]["available"])

            reused = service.create_job_from_existing_upload(
                upload_id=original.job_id,
                title="reused upload",
                metadata={"case": "reuse-target"},
                auto_start=True,
            )

            self.assertNotEqual(reused.job_id, original.job_id)
            self.assertEqual(store.get_job_current(reused.job_id)["job_status"], JobStatus.INGESTING)
            self.assertIn(("ingest", reused.job_id, None), queue.items)

            service.ingest_staged_upload(reused.job_id)

            self.assertEqual(store.get_job_current(reused.job_id)["job_status"], JobStatus.INPUT_READY)
            self.assertEqual(store.count_input_segments(reused.job_id), 1)
            reused_segment = store.list_input_segments(job_id=reused.job_id, include_content=True)[0]
            self.assertIn("first failure", reused_segment["content"])
            self.assertEqual([upload["upload_id"] for upload in store.list_staged_uploads()], [original.job_id])


if __name__ == "__main__":
    unittest.main()
