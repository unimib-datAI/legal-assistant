"""The project's single logging configuration point.

Library modules must only ever call ``logging.getLogger(__name__)``; configuring the
root logger is the *application's* job. Several modules used to call
``logging.basicConfig`` at import time, which meant importing the package silently
reconfigured logging for whoever imported it. Entry points — the CLI, the eval scripts,
the Streamlit app — call :func:`configure_logging` instead.
"""
from __future__ import annotations

import logging

FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
VERBOSE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO, *, show_logger_name: bool = False) -> None:
    """Configure root logging once. Safe to call again — later calls are no-ops.

    ``show_logger_name`` includes the module name in each line, which is what the
    ingestion pipelines want when several modules log interleaved progress.
    """
    logging.basicConfig(
        level=level,
        format=VERBOSE_FORMAT if show_logger_name else FORMAT,
        datefmt=DATE_FORMAT,
    )


def quiet(*logger_names: str, level: int = logging.WARNING) -> None:
    """Raise the level of chatty loggers, e.g. the per-node upserts during ingestion."""
    for name in logger_names:
        logging.getLogger(name).setLevel(level)
