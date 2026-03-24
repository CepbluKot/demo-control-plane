import logging
import sys
import threading

from .config import LOG_LEVEL, LOGS_DIR


def configure_logging() -> None:
    log_file = LOGS_DIR / "control-plane.log"
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s.%(msecs)03d | %(levelname)s | %(process)d:%(threadName)s | "
            "%(name)s:%(lineno)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.captureWarnings(True)
    control_plane_level = getattr(logging, LOG_LEVEL, logging.DEBUG)
    logging.getLogger("control_plane").setLevel(control_plane_level)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.INFO)
    logging.getLogger(__name__).info(
        "Logging configured: control_plane_level=%s, root_level=INFO, file=%s",
        LOG_LEVEL,
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
