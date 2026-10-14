"""Filesystem file reader adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_story.application.port.file_reader import FileReader

if TYPE_CHECKING:
    from pathlib import Path


class FilesystemFileReader(FileReader[str]):
    """Read a text file from the local filesystem."""

    TEXT_EXTENSIONS = frozenset({".txt", ".md", ".csv"})

    def read(self, path: Path) -> str:
        """Read one local text file as UTF-8.

        Compare the lowercased suffix with the supported text extensions.
        A match returns the file's UTF-8 text; any other suffix is refused.

        Args:
            path: File to read.

        Returns:
            The file's text.

        Raises:
            ValueError: The suffix is not a supported text extension.

        Flow:
            1. Return UTF-8 text for a supported suffix.
        """
        # 1. Return UTF-8 text for a supported suffix
        suffix = path.suffix.lower()
        if suffix in self.TEXT_EXTENSIONS:
            return path.read_text(encoding="utf-8")
        message = f"Unsupported file extension: {suffix}"
        raise ValueError(message)
