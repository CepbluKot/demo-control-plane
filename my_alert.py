from __future__ import annotations

from typing import Any, Dict, Optional


def send_sre_alert(
    *,
    summary_text: Optional[str] = None,
    summary: Optional[str] = None,
    anomaly: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Template alert adapter.
    Replace with real sender to Telegram/Slack/bot API/etc.

    Compatible with control_plane.processing adapter signatures.
    """
    text = summary_text or message or summary or ""
    anomaly_ts = (anomaly or {}).get("timestamp")

    # TODO: implement real delivery here.
    # Example:
    # requests.post(BOT_URL, json={"text": text}, timeout=10)
    return {
        "status": "template_sent",
        "sent": True,
        "channel": "template",
        "anomaly_timestamp": anomaly_ts,
        "message_preview": text[:200],
    }
