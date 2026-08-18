"""
Logging setup for the API process.

Call setup_logging() once, at startup, before anything else logs. Every other
module should just do:

    import logging
    logger = logging.getLogger(__name__)

so log lines carry the module path and we can raise/lower verbosity per package
without touching code (e.g. LEVELUP_LOG_LEVEL=DEBUG).
"""

import logging
import os
import sys

DEFAULT_LEVEL = os.getenv("LEVELUP_LOG_LEVEL", "INFO").upper()

# Timestamped, level-tagged, module-qualified. Plays well with `docker logs`
# and with HF Spaces / Cloud Run log viewers, which parse line-by-line.
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging for the process. Safe to call more than once."""
    root = logging.getLogger()
    if root.handlers:  # uvicorn may have configured things already
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root.addHandler(handler)
    root.setLevel(level or DEFAULT_LEVEL)

    # These are chatty at INFO and drown out our own lines.
    for noisy in ("httpx", "urllib3", "filelock", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
