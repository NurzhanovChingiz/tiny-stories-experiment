"""Acquire a dataset into raw storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_story.application.existing_file_policy import ExistingFilePolicy
from tiny_story.application.results.download_outcome import DownloadOutcome
from tiny_story.domain.entities.dataset_spec import require_dataset_spec
from tiny_story.domain.errors import (
    DownloadError,
    DownloadUnavailableError,
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
        existing: Whether a raw file whose SHA-256 matches the publication is kept
            or acquired again.
        run_logger: Run log for kept files, fetches, stores, and failures.

    Returns:
        The raw directory, the files stored now, and the files left in place.

    Flow:
        1. Reject a dataset spec that cannot be stored as raw files.
        2. Hold exclusive publication and select files that still need to be acquired.
        3. Store each acquired file and return the outcome.
    """
    # 1. Reject a dataset spec that cannot be stored as raw files.
    try:
        require_dataset_spec(spec)
    except (InvalidDatasetSpecError, InvalidRawFilenameError) as exc:
        _record(run_logger, str(exc), level="error")
        raise

    # 2. Hold exclusive publication and select files that still need to be acquired.
    _record(
        run_logger,
        (
            f"Acquiring {spec.name} with {len(spec.filenames)} file(s) "
            f"using {existing.value}"
        ),
    )
    with store.hold_publication():
        already_present, pending = _select_pending(
            spec,
            source,
            store,
            existing,
            run_logger,
        )
        for path in already_present:
            _record(run_logger, f"Keeping existing raw file {path}")

        # 3. Store each acquired file and return the outcome.
        written = [
            _acquire_pending(filename, digest, source, store, run_logger)
            for filename, digest in pending
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


def _select_pending(
    spec: DatasetSpec,
    source: DatasetSource,
    store: RawDatasetStore,
    existing: ExistingFilePolicy,
    run_logger: RunLogger | None,
) -> tuple[tuple[Path, ...], tuple[tuple[str, str], ...]]:
    """Select pending files and log a selection failure.

    Args:
        spec: Dataset identity.
        source: Retrieves the SHA-256 for each file.
        store: Raw storage used to see which files already match.
        existing: Whether a matching raw file is kept or acquired again.
        run_logger: Run log for a selection failure.

    Returns:
        Matching raw paths left in place, then each pending file name with its digest.

    Raises:
        DownloadError: A published digest could not be retrieved.
        InvalidRawFilenameError: A file name would leave the raw directory.
        RawStorageError: A stored file could not be inspected.

    Flow:
        1. Select files that still need to be acquired.
    """
    # 1. Select files that still need to be acquired.
    return _pending_filenames(spec, source, store, existing, run_logger)


def _pending_filenames(
    spec: DatasetSpec,
    source: DatasetSource,
    store: RawDatasetStore,
    existing: ExistingFilePolicy,
    run_logger: RunLogger | None,
) -> tuple[tuple[Path, ...], tuple[tuple[str, str], ...]]:
    """Return stored files to keep and file names still to acquire.

    Args:
        spec: Dataset identity.
        source: Retrieves the SHA-256 for each file.
        store: Raw storage used to see which files already match.
        existing: Whether a matching raw file is kept or acquired again.
        run_logger: Run log for a selection failure.

    Returns:
        Matching raw paths left in place, then each pending file name with its digest.

    Raises:
        DownloadError: A published digest could not be retrieved.
        InvalidRawFilenameError: A file name would leave the raw directory.
        RawStorageError: A stored file could not be inspected.

    Flow:
        1. In replace mode, pair every file with its published digest.
        2. Keep a copy that matches the published digest or this publication's record.
    """
    # 1. In replace mode, pair every file with its published digest.
    if existing == ExistingFilePolicy.REPLACE:
        return (), tuple(
            (filename, _published_digest(filename, source, run_logger))
            for filename in spec.filenames
        )

    # 2. Keep a copy that matches the published digest or this publication's record.
    present: list[Path] = []
    pending: list[tuple[str, str]] = []
    for filename in spec.filenames:
        kept = _kept_copy(filename, source, store, run_logger)
        if isinstance(kept, tuple):
            pending.append(kept)
        else:
            present.append(kept)
    return tuple(present), tuple(pending)


def _published_digest(
    filename: str,
    source: DatasetSource,
    run_logger: RunLogger | None,
) -> str:
    """Return one published digest and log a retrieval failure.

    Args:
        filename: Raw file name.
        source: Retrieves the published SHA-256.
        run_logger: Run log for a retrieval failure.

    Returns:
        The lowercase SHA-256 hex digest.

    Raises:
        DownloadError: The published digest could not be retrieved.
    """
    try:
        return source.content_sha256(filename)
    except (DownloadError, InvalidRawFilenameError, RawStorageError) as exc:
        _record_selection_failure(run_logger, filename, exc)
        raise


def _kept_copy(
    filename: str,
    source: DatasetSource,
    store: RawDatasetStore,
    run_logger: RunLogger | None,
) -> Path | tuple[str, str]:
    """Return a kept raw file, or the file name and digest still to acquire.

    Args:
        filename: Raw file name.
        source: Retrieves the published SHA-256.
        store: Raw storage used to see which file already matches.
        run_logger: Run log for a selection failure.

    Returns:
        The matching raw path, or the pending file name with its digest.

    Raises:
        DownloadError: The published digest could not be retrieved.
        DownloadUnavailableError: The Hub could not be reached and the store has
            no copy recorded for this publication.
        InvalidRawFilenameError: The name would leave the raw directory.
        RawStorageError: A stored file could not be inspected.

    Flow:
        1. Choose the kept copy or the digest still to acquire.
        2. Log and re-raise a download, filename, or storage error.
    """
    # 1. Choose the kept copy or the digest still to acquire.
    try:
        return _matching_copy(filename, source, store)
    except (DownloadError, InvalidRawFilenameError, RawStorageError) as exc:
        # 2. Log and re-raise a download, filename, or storage error.
        _record_selection_failure(run_logger, filename, exc)
        raise


def _matching_copy(
    filename: str,
    source: DatasetSource,
    store: RawDatasetStore,
) -> Path | tuple[str, str]:
    """Return a kept raw file, or the file name and digest still to acquire.

    Args:
        filename: Raw file name.
        source: Retrieves the published SHA-256.
        store: Raw storage used to see which file already matches.

    Returns:
        The matching raw path, or the pending file name with its digest.

    Raises:
        DownloadError: The published digest could not be retrieved.
        DownloadUnavailableError: The Hub could not be reached and the store has
            no copy recorded for this publication.

    Flow:
        1. Keep the copy recorded for this publication when the Hub cannot be reached.
        2. Keep a copy that matches the published digest, or return that digest.
    """
    # 1. Keep the copy recorded for this publication when the Hub cannot be reached.
    try:
        digest = source.content_sha256(filename)
    except DownloadUnavailableError:
        found = store.verified_file(filename, source.publication_id)
        if found is None:
            raise
        return found

    # 2. Keep a copy that matches the published digest, or return that digest.
    found = store.existing_file(filename, digest, source.publication_id)
    if found is None:
        return (filename, digest)
    return found


def _record_selection_failure(
    run_logger: RunLogger | None,
    filename: str,
    exc: Exception,
) -> None:
    """Record one file that could not be selected.

    Args:
        run_logger: Run log, or None when this acquisition is not recorded.
        filename: Raw file name that could not be selected.
        exc: Download, filename, or storage failure.

    Returns:
        None.
    """
    _record(run_logger, f"Acquisition failed for {filename}: {exc}", level="error")


def _acquire_pending(
    filename: str,
    content_sha256: str,
    source: DatasetSource,
    store: RawDatasetStore,
    run_logger: RunLogger | None,
) -> Path:
    """Fetch one pending file, store it, and record the attempt.

    Args:
        filename: Raw file name to acquire.
        content_sha256: SHA-256 the stored copy must match.
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
        stored = store.store_unchanged(
            filename,
            source.fetch(filename),
            content_sha256,
            source.publication_id,
        )
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
