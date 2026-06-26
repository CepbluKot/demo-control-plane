from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient

from summary_frontend.app import app


class SummaryFrontendTests(unittest.TestCase):
    def test_index_and_dynamic_config(self) -> None:
        old_http = os.environ.get("SUMMARY_FRONTEND_BACKEND_HTTP_URL")
        old_ws = os.environ.get("SUMMARY_FRONTEND_BACKEND_WS_URL")
        os.environ["SUMMARY_FRONTEND_BACKEND_HTTP_URL"] = "http://backend.example:8088"
        os.environ["SUMMARY_FRONTEND_BACKEND_WS_URL"] = "ws://backend.example:8088"
        try:
            client = TestClient(app)

            index = client.get("/")
            config = client.get("/config.js")

            self.assertEqual(index.status_code, 200)
            self.assertIn("Summary Jobs", index.text)
            self.assertEqual(config.status_code, 200)
            self.assertIn("http://backend.example:8088", config.text)
            self.assertIn("ws://backend.example:8088", config.text)
        finally:
            if old_http is None:
                os.environ.pop("SUMMARY_FRONTEND_BACKEND_HTTP_URL", None)
            else:
                os.environ["SUMMARY_FRONTEND_BACKEND_HTTP_URL"] = old_http
            if old_ws is None:
                os.environ.pop("SUMMARY_FRONTEND_BACKEND_WS_URL", None)
            else:
                os.environ["SUMMARY_FRONTEND_BACKEND_WS_URL"] = old_ws


if __name__ == "__main__":
    unittest.main()
