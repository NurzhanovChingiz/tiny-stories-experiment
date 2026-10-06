"""Place published dataset files into raw storage without replacing them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tiny_stories_experiment.application.results.download_outcome import DownloadOutcome
from tiny_stories_experiment.domain.errors import (
    DownloadSizeMismatchError,
    RawFileConflictError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tiny_stories_experiment.application.ports.published_dataset_source import (
        PublishedDatasetSource,
    )
    from tiny_stories_experiment.application.ports.raw_dataset_store import (
        RawDatasetStore,
    )
    from tiny_stories_experiment.domain.dataset.dataset_spec import DatasetSpec


def place_published_file(
    destination: Path,
    filename: str,
    source: PublishedDatasetSource,
    store: RawDatasetStore,
) -> Literal["kept", "downloaded"]:
    """Leave a matching raw file in place, or download it when it is absent.

    A local file whose size differs from the published copy stays untouched.
    Remove that file before running the download again.

    Args:
        destination: Directory that stores immutable raw files.
        filename: Published file name to place.
        source: Remote catalog that knows the published byte count.
        store: Local raw-file storage.

    Returns:
        ``kept`` when the local file already matches the published size,
        otherwise ``downloaded``.

    Raises:
        RawFileConflictError: The local file exists and its size differs.
        DownloadSizeMismatchError: The fetched file size differs from the
            published byte count.

    Flow:
        1. Published size — read the remote byte count.
        2. Local size check — keep a match and refuse a different local file.
        3. Missing-file download — fetch the file when no local copy exists.
    """
    # 1. Published size
    remote_size = source.byte_count(filename)
    local_size = store.byte_count(destination, filename)

    # 2. Local size check
    if local_size is not None:
        if local_size == remote_size:
            return "kept"
        message = (
            f"Raw file {destination / filename} is {local_size} bytes; "
            f"published file {filename} is {remote_size} bytes. "
            "Remove the local file before downloading again."
        )
        raise RawFileConflictError(message)

    # 3. Missing-file download
    source.fetch(filename, destination)
    downloaded_size = store.byte_count(destination, filename)
    if downloaded_size != remote_size:
        message = (
            f"Downloaded {destination / filename} is {downloaded_size} bytes; "
            f"published file {filename} is {remote_size} bytes."
        )
        raise DownloadSizeMismatchError(message)
    return "downloaded"


def download_dataset(
    spec: DatasetSpec,
    destination: Path,
    source: PublishedDatasetSource,
    store: RawDatasetStore,
) -> DownloadOutcome:
    """Place each file in a dataset spec into a raw directory.

    Args:
        spec: Published dataset to download.
        destination: Directory for the immutable raw files.
        source: Remote catalog bound to that dataset.
        store: Local raw-file storage.

    Returns:
        Which files were already present and which were fetched.

    Flow:
        1. Raw directory — create the destination when it is missing.
        2. Place each file — keep a size match or download a missing file.
        3. Placement report — return the kept and downloaded names.
    """
    # 1. Raw directory
    store.prepare(destination)
    kept: list[str] = []
    downloaded: list[str] = []

    # 2. Place each file
    for filename in spec.filenames:
        placement = place_published_file(destination, filename, source, store)
        if placement == "kept":
            kept.append(filename)
        else:
            downloaded.append(filename)

    # 3. Placement report
    return DownloadOutcome(
        destination=destination,
        kept=tuple(kept),
        downloaded=tuple(downloaded),
    )
