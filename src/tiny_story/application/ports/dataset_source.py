"""Acquisition contract for one configured dataset source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class DatasetSource(ABC):
    """Retrieves one raw dataset file without changing its bytes."""

    @property
    @abstractmethod
    def publication_id(self) -> str:
        """Return the stable identity of this repository revision.

        Returns:
            A stable identity for one repository and revision.

        Raises:
            DownloadError: The publication identity could not be determined.
            DownloadUnavailableError: The Hub could not be reached.
        """

    @abstractmethod
    def fetch(self, filename: str) -> Path:
        """Acquire one configured file and return its local path.

        The file at the returned path must be left unchanged.

        Args:
            filename: Raw file name requested for this dataset.

        Returns:
            Path of the acquired file.

        Raises:
            DownloadError: The file could not be retrieved.
            DownloadUnavailableError: The Hub could not be reached.
        """

    @abstractmethod
    def content_sha256(self, filename: str) -> str:
        """Return the lowercase SHA-256 of one published file.

        Args:
            filename: Raw file name whose published bytes are hashed.

        Returns:
            The lowercase SHA-256 hex digest.

        Raises:
            DownloadError: The digest could not be retrieved.
            DownloadUnavailableError: The Hub could not be reached.
        """
