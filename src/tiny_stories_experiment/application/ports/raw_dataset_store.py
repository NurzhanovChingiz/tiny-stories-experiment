"""Port for immutable local raw dataset files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class RawDatasetStore(ABC):
    """Local storage for immutable raw dataset files."""

    @abstractmethod
    def prepare(self, destination: Path) -> None:
        """Create the raw directory when it is missing.

        Args:
            destination: Directory that will hold raw files.
        """

    @abstractmethod
    def byte_count(self, destination: Path, filename: str) -> int | None:
        """Return the local file size, or ``None`` when the file is absent.

        Args:
            destination: Directory that holds raw files.
            filename: File name inside that directory.

        Returns:
            The local size in bytes, or ``None`` when the file is absent.
        """
