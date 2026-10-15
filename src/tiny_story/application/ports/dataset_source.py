"""Acquisition contract for one configured dataset source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class DatasetSource(ABC):
    """Retrieves one raw dataset file without changing its bytes."""

    @abstractmethod
    def fetch(self, filename: str) -> Path:
        """Acquire one configured file and return its local path.

        The file at the returned path must be left unchanged.

        Args:
            filename: Raw file name requested for this dataset.

        Returns:
            Path of the acquired file.
        """
