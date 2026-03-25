import logging
import sys
import threading
from logging.handlers import RotatingFileHandler

from .config import LOGS_DIR


class _AllowedLevelsFilter(logging.Filter):
    def __init__(self, allowed_levels: set[int]) -> None:
        super().__init__()
        self.allowed_levels = allowed_levels

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self.allowed_levels


def configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "control-plane.log"
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    log_format = (
        "%(asctime)s.%(msecs)03d | %(levelname)s | %(process)d:%(threadName)s | "
        "%(name)s:%(lineno)d | %(message)s"
    )
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    main_handler = RotatingFileHandler(
        log_file,
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    # Only INFO/WARNING/ERROR go to one file: control-plane.log
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(formatter)
    main_handler.addFilter(
        _AllowedLevelsFilter({logging.INFO, logging.WARNING, logging.ERROR})
    )

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(main_handler)

    logging.captureWarnings(True)
    logging.getLogger("control_plane").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.INFO)
    logging.getLogger(__name__).info(
        "Logging configured: levels=INFO|WARNING|ERROR, file=%s",
        log_file,
    )

    def _handle_uncaught(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("uncaught").exception(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def _handle_thread_exception(args):
        logging.getLogger("uncaught.thread").exception(
            "Uncaught thread exception",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _handle_uncaught
    threading.excepthook = _handle_thread_exception
