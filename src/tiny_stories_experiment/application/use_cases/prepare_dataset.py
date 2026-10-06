"""Write derived story text from raw JSONL files without changing those files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_stories_experiment.application.ports.processed_text_store import (
    processed_filename,
)
from tiny_stories_experiment.application.results.prepare_outcome import (
    PreparedSplit,
    PrepareOutcome,
)
from tiny_stories_experiment.domain.dataset.text_sample import split_from_filename
from tiny_stories_experiment.domain.errors import (
    RawDatasetChangedError,
    RawDatasetMissingError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tiny_stories_experiment.application.ports.processed_text_store import (
        ProcessedTextStore,
    )
    from tiny_stories_experiment.application.ports.raw_dataset_store import (
        RawDatasetStore,
    )
    from tiny_stories_experiment.application.ports.raw_story_source import (
        RawStorySource,
    )
    from tiny_stories_experiment.domain.dataset.dataset_spec import DatasetSpec


def prepare_dataset(
    spec: DatasetSpec,
    raw_directory: Path,
    processed_directory: Path,
    *,
    source: RawStorySource,
    processed_store: ProcessedTextStore,
    raw_store: RawDatasetStore,
) -> PrepareOutcome:
    """Write one derived JSONL file per raw split and leave the raw files unchanged.

    A failed write still checks that the raw byte counts are the ones recorded
    at the start. Derived files may be replaced; raw files may not.

    Args:
        spec: Dataset whose raw file names are prepared.
        raw_directory: Directory of immutable raw JSONL files.
        processed_directory: Directory for derived ``train.jsonl`` and
            ``valid.jsonl`` files.
        source: Reader for story text in one raw JSONL file.
        processed_store: Writer for one split's derived JSONL file.
        raw_store: Size checks for the raw files.

    Returns:
        The split, story count, and derived file name for each raw file.

    Raises:
        RawDatasetMissingError: A raw file named by the spec is absent.
        UnknownDatasetSplitError: A raw file name does not end in a published
            split token.
        RawStoryRecordError: A raw line is not an object with a string text
            field.
        RawDatasetChangedError: A raw file's size changed during preparation.

    Flow:
        1. Source sizes — record each raw file's byte count.
        2. Processed directory — create the destination when it is missing.
        3. Write each split — stream that file's stories into its derived JSONL.
        4. Raw unchanged — refuse the result when a recorded size no longer matches.
        5. Preparation report — return each split's count and output name.
    """
    # 1. Source sizes
    sizes = _raw_sizes(spec, raw_directory, raw_store)

    # 2. Processed directory
    processed_store.prepare(processed_directory)

    # 3. Write each split
    try:
        splits = tuple(
            _write_split(
                raw_directory,
                processed_directory,
                filename,
                source,
                processed_store,
            )
            for filename in spec.filenames
        )
    finally:
        # 4. Raw unchanged
        _require_raw_unchanged(raw_directory, raw_store, sizes)

    # 5. Preparation report
    return PrepareOutcome(
        raw_directory=raw_directory,
        processed_directory=processed_directory,
        splits=splits,
    )


def _raw_sizes(
    spec: DatasetSpec,
    raw_directory: Path,
    raw_store: RawDatasetStore,
) -> dict[str, int]:
    """Record the byte count of each raw file named by the spec.

    Args:
        spec: Dataset whose raw file names are required.
        raw_directory: Directory of immutable raw JSONL files.
        raw_store: Size checks for the raw files.

    Returns:
        Byte count by raw file name.

    Raises:
        RawDatasetMissingError: A named raw file is absent.

    Flow:
        1. Local size — read the byte count of each named raw file.
        2. Missing file — refuse when a named file is absent.
    """
    sizes: dict[str, int] = {}
    for filename in spec.filenames:
        # 1. Local size
        size = raw_store.byte_count(raw_directory, filename)
        # 2. Missing file
        if size is None:
            message = f"Raw file {raw_directory / filename} is absent."
            raise RawDatasetMissingError(message)
        sizes[filename] = size
    return sizes


def _write_split(
    raw_directory: Path,
    processed_directory: Path,
    filename: str,
    source: RawStorySource,
    processed_store: ProcessedTextStore,
) -> PreparedSplit:
    """Stream one raw file into that split's derived JSONL.

    Args:
        raw_directory: Directory of immutable raw JSONL files.
        processed_directory: Directory for derived JSONL files.
        filename: Raw file name to read.
        source: Reader for story text in one raw JSONL file.
        processed_store: Writer for one split's derived JSONL file.

    Returns:
        The split, story count, and derived file name.

    Flow:
        1. Split name — take the published split token from the raw file name.
        2. Story text — stream records from that raw file.
        3. Derived file — write those stories and return the count.
    """
    # 1. Split name
    split = split_from_filename(filename)

    # 2. Story text
    texts = source.texts(raw_directory / filename)

    # 3. Derived file
    story_count = processed_store.write(processed_directory, split, texts)
    return PreparedSplit(
        raw_filename=filename,
        split=split,
        story_count=story_count,
        output_name=processed_filename(split),
    )


def _require_raw_unchanged(
    raw_directory: Path,
    raw_store: RawDatasetStore,
    sizes: dict[str, int],
) -> None:
    """Refuse preparation when a raw file no longer has its recorded size.

    Args:
        raw_directory: Directory of immutable raw JSONL files.
        raw_store: Size checks for the raw files.
        sizes: Byte count by raw file name recorded before writing.

    Raises:
        RawDatasetChangedError: A raw file's size differs from the recorded count.
    """
    for filename, size in sizes.items():
        current = raw_store.byte_count(raw_directory, filename)
        if current != size:
            message = (
                f"Raw file {raw_directory / filename} changed from {size} "
                f"bytes to {current} bytes while preparing derived text."
            )
            raise RawDatasetChangedError(message)
