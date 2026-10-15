"""Acquire a dataset into raw storage."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from tiny_story.application.results.download_outcome import DownloadOutcome
from tiny_story.domain.dataset.dataset_spec import require_dataset_spec

if TYPE_CHECKING:
    from pathlib import Path

    from tiny_story.application.ports.dataset_source import DatasetSource
    from tiny_story.application.ports.raw_dataset_store import RawDatasetStore
    from tiny_story.domain.dataset.dataset_spec import DatasetSpec


class ExistingFilePolicy(StrEnum):
    """What to do when a non-empty raw file is already stored."""

    KEEP = "keep"
    REPLACE = "replace"


def download_dataset(
    spec: DatasetSpec,
    source: DatasetSource,
    store: RawDatasetStore,
    existing: ExistingFilePolicy = ExistingFilePolicy.KEEP,
) -> DownloadOutcome:
    """Acquire missing raw files and record files left in place.

    Args:
        spec: Dataset identity, including the file names to store.
        source: Retrieves one configured file at a time.
        store: Holds unmodified raw copies.
        existing: Whether a non-empty raw file is kept or acquired again.

    Returns:
        The raw directory, the files stored now, and the files left in place.

    Flow:
        1. Reject a dataset spec that cannot be stored as raw files.
        2. Select files that still need to be acquired.
        3. Store each acquired file and return the outcome.
    """
    # 1. Reject a dataset spec that cannot be stored as raw files.
    require_dataset_spec(spec)

    # 2. Select files that still need to be acquired.
    already_present, pending = _pending_filenames(spec, store, existing)

    # 3. Store each acquired file and return the outcome.
    written = [
        store.store_unchanged(filename, source.fetch(filename)) for filename in pending
    ]
    return DownloadOutcome(
        directory=store.directory,
        written_paths=tuple(written),
        already_present_paths=already_present,
    )


def _pending_filenames(
    spec: DatasetSpec,
    store: RawDatasetStore,
    existing: ExistingFilePolicy,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Return stored files to keep and file names still to acquire.

    Args:
        spec: Dataset identity.
        store: Raw storage used to see which files are already complete.
        existing: Whether a non-empty raw file is kept or acquired again.

    Returns:
        Complete raw paths left in place, then file names to acquire.
    """
    if existing is ExistingFilePolicy.REPLACE:
        return (), spec.filenames
    present: list[Path] = []
    pending: list[str] = []
    for filename in spec.filenames:
        found = store.existing_file(filename)
        if found is None:
            pending.append(filename)
        else:
            present.append(found)
    return tuple(present), tuple(pending)
