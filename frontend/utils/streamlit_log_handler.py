import logging

import streamlit as st


class StreamlitLogHandler(logging.Handler):
    """Routes Python log records into a Streamlit text area."""

    def __init__(self, container):
        super().__init__()
        self._lines: list[str] = []
        self._container = container
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        self._lines.append(self.format(record))
        self._container.text_area("Output", "\n".join(self._lines), height=200)
