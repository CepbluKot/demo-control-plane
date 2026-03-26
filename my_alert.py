from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def send_sre_alert(text: str) -> Dict[str, Any]:
    """
    Template alert adapter.
    Replace with real sender to Telegram/Slack/bot API/etc.
    """

    # TODO: implement real delivery here.
    # Example:
    # requests.post(BOT_URL, json={"text": text}, timeout=10)
    logger.info(
        "my_alert.send_sre_alert called: text_len=%s preview=%s",
        len(text),
        text[:120],
    )
    return {
        "status": "template_sent",
        "sent": True,
        "channel": "template",
        "sender": "my_alert.send_sre_alert",
        "message_preview": text[:200],
    }
