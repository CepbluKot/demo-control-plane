from __future__ import annotations

from typing import Dict


def send_sre_alert(text: str) -> Dict[str, Any]:
    """
    Template alert adapter.
    Replace with real sender to Telegram/Slack/bot API/etc.
    """

    # TODO: implement real delivery here.
    # Example:
    # requests.post(BOT_URL, json={"text": text}, timeout=10)
    return {
        "status": "template_sent",
        "sent": True,
        "channel": "template",
        "message_preview": text[:200],
    }
