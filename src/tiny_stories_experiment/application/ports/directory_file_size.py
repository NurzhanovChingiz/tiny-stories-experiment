"""Port for the byte count of one file in a directory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class DirectoryFileSize(Protocol):
    """Read-only size check for a named file in a directory."""

    def byte_count(self, destination: Path, filename: str) -> int | None:
        """Return the file size, or ``None`` when the file is absent.

        Args:
            destination: Directory that holds the file.
            filename: File name inside that directory.

        Returns:
            The size in bytes, or ``None`` when the file is absent.
        """
