"""Shared logging utility for the AI Secure Agent."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s %(levelname)s [%(name)s] "
                "[service_name=ai-secure-agent-ms] "
                # "[trace_id=%(trace_id)s] - %(message)s"
                " - %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
