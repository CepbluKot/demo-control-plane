import json
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.ch_client import ch_query


def _mock_resp(rows):
    body = "\n".join(json.dumps(r) for r in rows).encode()
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    m.read.return_value = body
    return m


def test_parses_jsoneachrow():
    rows = [{"timestamp": "2024-01-01", "raw_line": "foo"}]
    with patch("urllib.request.urlopen", return_value=_mock_resp(rows)):
        assert ch_query("localhost", 8123, "u", "p", "SELECT 1") == rows


def test_skips_empty_lines():
    body = b'{"a": 1}\n\n{"a": 2}\n'
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    m.read.return_value = body
    with patch("urllib.request.urlopen", return_value=m):
        assert len(ch_query("localhost", 8123, "u", "p", "SELECT 1")) == 2
