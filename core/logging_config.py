"""Logging setup: stdlib ``logging`` with a rotating file handler (§1).

Configured exactly once, from the CLI entry point. Library modules just call
``logging.getLogger(__name__)`` and never touch handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from core.config import REPO_ROOT, get_settings

_configured = False


def setup_logging(level: str | None = None, force: bool = False) -> None:
    """Install console + rotating-file handlers on the root logger.

    Idempotent: a second call is a no-op unless ``force`` is set. This matters
    because the orchestrator may be restarted mid-day (§9.5) inside the same
    process during tests.
    """
    global _configured
    if _configured and not force:
        return

    settings = get_settings()
    log_level = (level or os.environ.get("LOG_LEVEL") or settings.get("logging.level", "INFO")).upper()
    log_dir = REPO_ROOT / str(settings.get("logging.dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / str(settings.get("logging.filename", "trading.log"))

    fmt = logging.Formatter(
        settings.get("logging.format", "%(asctime)s %(levelname)-8s %(name)-22s %(message)s")
    )

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
    root.setLevel(log_level)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=int(settings.get("logging.max_bytes", 10_485_760)),
        backupCount=int(settings.get("logging.backup_count", 10)),
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    # These libraries are chatty at DEBUG and drown our own lines.
    for noisy in ("httpx", "httpcore", "apscheduler.executors.default", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor so modules need one import, not two."""
    return logging.getLogger(name)


def log_path() -> Path:
    """Absolute path of the active log file -- printed by ``--status``."""
    settings = get_settings()
    return REPO_ROOT / str(settings.get("logging.dir", "logs")) / str(
        settings.get("logging.filename", "trading.log")
    )
