from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def make_alert(text: str) -> Dict[str, Any]:
    """
    Заглушка интерфейса отправки alert.
    Пока ничего не отправляет, только возвращает технический результат.
    """
    logger.info(
        "alerts.make_alert.stub called: text_len=%s",
        len(text),
    )
    return {
        "status": "stub",
        "sent": False,
        "message": (
            "Alert sender is not configured yet. "
            "Set CONTROL_PLANE_ALERT_CALLABLE or implement control_plane.alerts.make_alert."
        ),
        "summary_preview": text[:160],
    }
