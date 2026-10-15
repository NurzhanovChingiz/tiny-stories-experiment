"""Result of storing an acquired dataset as raw files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class DownloadOutcome:
    """Files written by a download and files that were already stored.

    Attributes:
        directory: Raw directory that holds the dataset files.
        written_paths: Raw files stored during this acquisition.
        already_present_paths: Raw files left untouched because they were already complete.
    """

    directory: Path
    written_paths: tuple[Path, ...]
    already_present_paths: tuple[Path, ...]
