"""Raw storage contract for unmodified acquired dataset files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class RawDatasetStore(ABC):
    """Stores acquired files as raw inputs."""

    @property
    @abstractmethod
    def directory(self) -> Path:
        """Directory that receives unmodified raw files.

        Returns:
            The raw directory.
        """

    @abstractmethod
    def existing_file(self, filename: str) -> Path | None:
        """Return the stored file when it is already present.

        Args:
            filename: Raw file name.

        Returns:
            The raw path, or None when that file is absent.
        """

    @abstractmethod
    def store_unchanged(self, filename: str, source_file: Path) -> Path:
        """Copy acquired bytes into raw storage.

        Args:
            filename: Raw file name.
            source_file: Local file whose bytes are copied.

        Returns:
            The raw path that now holds those bytes.
        """
