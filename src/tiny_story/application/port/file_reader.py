"""FileReader gateway — reads a file into a typed value."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class FileReader[T](ABC):
    """Port for reading a file."""

    @abstractmethod
    def read(self, path: Path) -> T:
        """Read a file and return its content.

        Args:
            path: File to read.

        Returns:
            The file content.
        """
