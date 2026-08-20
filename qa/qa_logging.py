import logging
import sys

from . import qa_config

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, str(qa_config.LOG_LEVEL).upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger("qa")
    root.setLevel(level)
    # Tránh add handler trùng lặp nếu module bị import lại (vd. trong test).
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Trả về logger con dưới namespace 'qa', đã cấu hình sẵn handler/level."""
    _configure_root()
    return logging.getLogger(f"qa.{name}")
