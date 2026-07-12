"""Central logging configuration.

Call :func:`configure_logging` once at process start (Streamlit entrypoint,
CLI, tests). Modules should just call ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging

from config.settings import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    _CONFIGURED = True
