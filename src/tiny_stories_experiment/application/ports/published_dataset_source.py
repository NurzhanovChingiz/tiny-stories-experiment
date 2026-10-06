"""Port for reading and fetching a published dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class PublishedDatasetSource(ABC):
    """Remote catalog of one published dataset."""

    @abstractmethod
    def byte_count(self, filename: str) -> int:
        """Return the published size of one file in bytes.

        Args:
            filename: File name inside the published dataset.

        Returns:
            The published file size in bytes.
        """

    @abstractmethod
    def fetch(self, filename: str, destination: Path) -> None:
        """Copy one published file into a local directory.

        Args:
            filename: File name inside the published dataset.
            destination: Directory that receives the file.
        """
