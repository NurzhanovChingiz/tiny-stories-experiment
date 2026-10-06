"""Result of placing published dataset files in raw storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class DownloadOutcome:
    """Files kept or downloaded into the raw directory.

    Attributes:
        destination: Directory that holds the raw files.
        kept: File names already present with the published byte count.
        downloaded: File names fetched during this call.
    """

    destination: Path
    kept: tuple[str, ...]
    downloaded: tuple[str, ...]
