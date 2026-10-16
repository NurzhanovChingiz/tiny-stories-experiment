"""Acquire a dataset into raw storage."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from tiny_story.application.results.download_outcome import DownloadOutcome
from tiny_story.domain.entities.dataset_spec import require_dataset_spec
from tiny_story.domain.errors import (
    DownloadError,
    InvalidDatasetSpecError,
    InvalidRawFilenameError,
    RawStorageError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tiny_story.application.ports.dataset_source import DatasetSource
    from tiny_story.application.ports.raw_dataset_store import RawDatasetStore
    from tiny_story.application.ports.run_logger import RunLogger
    from tiny_story.domain.entities.dataset_spec import DatasetSpec


class ExistingFilePolicy(StrEnum):
    """What to do when a non-empty raw file is already stored."""

    KEEP = "keep"
    REPLACE = "replace"


def download_dataset(
    spec: DatasetSpec,
    source: DatasetSource,
    store: RawDatasetStore,
    existing: ExistingFilePolicy = ExistingFilePolicy.KEEP,
    run_logger: RunLogger | None = None,
) -> DownloadOutcome:
    """Acquire missing raw files and record files left in place.

    Args:
        spec: Dataset identity, including the file names to store.
        source: Retrieves one configured file at a time.
        store: Holds unmodified raw copies.
        existing: Whether a non-empty raw file is kept or acquired again.
        run_logger: Run log for kept files, fetches, stores, and failures.

    Returns:
        The raw directory, the files stored now, and the files left in place.

    Flow:
        1. Reject a dataset spec that cannot be stored as raw files.
        2. Select files that still need to be acquired.
        3. Store each acquired file and return the outcome.
    """
    # 1. Reject a dataset spec that cannot be stored as raw files.
    try:
        require_dataset_spec(spec)
    except (InvalidDatasetSpecError, InvalidRawFilenameError) as exc:
        _record(run_logger, str(exc), level="error")
        raise

    # 2. Select files that still need to be acquired.
    _record(
        run_logger,
        (
            f"Acquiring {spec.name} with {len(spec.filenames)} file(s) "
            f"using {existing.value}"
        ),
    )
    already_present, pending = _pending_filenames(spec, store, existing)
    for path in already_present:
        _record(run_logger, f"Keeping existing raw file {path}")

    # 3. Store each acquired file and return the outcome.
    written = [
        _acquire_pending(filename, source, store, run_logger) for filename in pending
    ]
    outcome = DownloadOutcome(
        directory=store.directory,
        written_paths=tuple(written),
        already_present_paths=already_present,
    )
    _record(
        run_logger,
        (
            f"Acquired {spec.name}: wrote {len(outcome.written_paths)} file(s), "
            f"kept {len(outcome.already_present_paths)} file(s) "
            f"in {outcome.directory}"
        ),
    )
    return outcome


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


def _acquire_pending(
    filename: str,
    source: DatasetSource,
    store: RawDatasetStore,
    run_logger: RunLogger | None,
) -> Path:
    """Fetch one pending file, store it, and record the attempt.

    Args:
        filename: Raw file name to acquire.
        source: Retrieves the configured file.
        store: Holds the unmodified raw copy.
        run_logger: Run log for the fetch, the stored path, or the failure.

    Returns:
        The raw path that now holds the file.

    Raises:
        DownloadError: The file could not be retrieved.
        RawStorageError: The bytes could not be stored.
        InvalidRawFilenameError: The name would leave the raw directory.
    """
    _record(run_logger, f"Fetching {filename}")
    try:
        stored = store.store_unchanged(filename, source.fetch(filename))
    except (DownloadError, InvalidRawFilenameError, RawStorageError) as exc:
        _record(run_logger, f"Acquisition failed for {filename}: {exc}", level="error")
        raise
    _record(run_logger, f"Stored raw file {stored}")
    return stored


def _record(
    run_logger: RunLogger | None,
    message: str,
    *,
    level: str = "info",
) -> None:
    """Record one run fact when a logger is attached.

    Args:
        run_logger: Run log, or None when this acquisition is not recorded.
        message: Fact to record.
        level: Severity name accepted by the run logger.

    Returns:
        None.
    """
    if run_logger is not None:
        run_logger.log(message, level=level)
