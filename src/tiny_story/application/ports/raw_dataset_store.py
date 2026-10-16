"""Raw storage contract for unmodified acquired dataset files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextlib import AbstractContextManager
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
    def hold_publication(self) -> AbstractContextManager[None]:
        """Hold exclusive publication across every file in one acquisition.

        Returns:
            A context manager that keeps publication exclusive until it exits.
            File operations on this store reuse that hold instead of locking again.
        """

    @abstractmethod
    def existing_file(
        self,
        filename: str,
        content_sha256: str,
        publication_id: str,
    ) -> Path | None:
        """Return the stored file when its bytes match a published digest.

        Args:
            filename: Raw file name.
            content_sha256: Lowercase SHA-256 hex digest of the published bytes.
            publication_id: Publication whose digest is being recorded.

        Returns:
            The raw path, or None when that file is absent or has a different digest.

        A match is recorded with ``publication_id``. A mismatch clears that
        file's publication record. A missing file leaves any record alone.
        """

    @abstractmethod
    def verified_file(self, filename: str, publication_id: str) -> Path | None:
        """Return the stored file when it still matches one publication's record.

        Args:
            filename: Raw file name.
            publication_id: Publication the recorded digest must name.

        Returns:
            The raw path, or None when that file is absent, has no record for
            ``publication_id``, or no longer matches the recorded digest.
        """

    @abstractmethod
    def store_unchanged(
        self,
        filename: str,
        source_file: Path,
        content_sha256: str,
        publication_id: str,
    ) -> Path:
        """Copy acquired bytes into raw storage when they match a digest.

        Args:
            filename: Raw file name.
            source_file: Local file whose bytes are copied.
            content_sha256: Lowercase SHA-256 hex digest the copy must match.
            publication_id: Publication recorded with the verified digest.

        Returns:
            The raw path that now holds those bytes.
        """
