"""Filesystem run logger adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from tiny_story.application.ports.run_logger import RunLogger

if TYPE_CHECKING:
    from pathlib import Path


class FilesystemRunLogger(RunLogger):
    """Logs run messages to a file."""

    def __init__(self, log_path: Path) -> None:
        """Attach an isolated file sink for this run logger.

        Create the destination directory, bind a unique logger identity, and
        add a file handler that accepts only that identity.

        Args:
            log_path: File that receives this adapter's records.

        Returns:
            None.

        Flow:
            1. Ensure the log directory exists.
            2. Bind an isolated logger identity.
            3. Attach a filtered file sink.
        """
        # 1. Ensure the log directory exists.
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Bind an isolated logger identity.
        self._logger_id = str(uuid4())

        self._logger = logger.bind(
            run_logger_id=self._logger_id,
        )

        # 3. Attach a filtered file sink.
        self._handler_id = logger.add(
            self.log_path,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="10 MB",
            filter=lambda record: (
                record["extra"].get("run_logger_id") == self._logger_id
            ),
        )

    def log(self, message: str, *, level: str = "info") -> None:
        """Log a message with the specified severity."""
        self._logger.log(level.upper(), message)

    def close(self) -> None:
        """Remove this adapter's logging handler."""
        logger.remove(self._handler_id)
