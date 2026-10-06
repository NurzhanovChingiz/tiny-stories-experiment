"""Filesystem adapter for immutable raw dataset files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_stories_experiment.application.ports.raw_dataset_store import RawDatasetStore

if TYPE_CHECKING:
    from pathlib import Path


class FilesystemDatasetRepository(RawDatasetStore):
    """Raw dataset files stored as ordinary files on disk."""

    def prepare(self, destination: Path) -> None:
        """Create the raw directory when it is missing.

        Args:
            destination: Directory that will hold raw files.
        """
        destination.mkdir(parents=True, exist_ok=True)

    def byte_count(self, destination: Path, filename: str) -> int | None:
        """Return the local file size, or ``None`` when the file is absent.

        Args:
            destination: Directory that holds raw files.
            filename: File name inside that directory.

        Returns:
            The local size in bytes, or ``None`` when the file is absent.

        Flow:
            1. Absent file — return None when the path is not a file.
            2. Local size — return the byte count of the existing file.
        """
        # 1. Absent file
        path = destination / filename
        if not path.is_file():
            return None
        # 2. Local size
        return path.stat().st_size

    def remove(self, destination: Path, filename: str) -> None:
        """Delete one local raw file when it is present.

        Args:
            destination: Directory that holds raw files.
            filename: File name inside that directory.
        """
        (destination / filename).unlink(missing_ok=True)
