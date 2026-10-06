"""Write derived story text from raw JSONL files without changing those files."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

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
    RawStoryRecordError,
    UnknownDatasetSplitError,
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

_STAGE_ERRORS = (OSError, RawStoryRecordError, UnknownDatasetSplitError)


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

    Derived files are replaced only after every split has been staged. A failed
    stage or a raw size change leaves the previous derived files in place.
    When staging fails and a raw size also changes, the stage error is raised
    and the size error is its cause.

    Args:
        spec: Dataset whose raw file names are prepared.
        raw_directory: Directory of immutable raw JSONL files.
        processed_directory: Directory for derived ``train.jsonl`` and
            ``valid.jsonl`` files.
        source: Reader for story text in one raw JSONL file.
        processed_store: Writer for staged and published split files.
        raw_store: Size checks for the raw files.

    Returns:
        The split, story count, and derived file name for each raw file.

    Raises:
        RawDatasetMissingError: A raw file named by the spec is absent.
        UnknownDatasetSplitError: A raw file name does not end in a published
            split token.
        RawStoryRecordError: A raw line is not an object with a string text
            field. A raw size change during that failure is attached as the
            cause.
        RawDatasetChangedError: Staging succeeded and a raw file's size changed.
        OSError: Staging or publishing a derived file failed on disk.

    Flow:
        1. Source sizes — record each raw file's byte count.
        2. Processed directory — create the destination when it is missing.
        3. Stage each split — stream that file's stories into a partial file.
        4. Raw unchanged — read the recorded sizes again after staging.
        5. Publish or refuse — replace derived files, or drop partials and raise.
        6. Preparation report — return each split's count and output name.
    """
    # 1. Source sizes
    sizes = _raw_sizes(spec, raw_directory, raw_store)

    # 2. Processed directory
    processed_store.prepare(processed_directory)

    # 3. Stage each split
    staged, stage_error = _stage_splits(
        spec,
        raw_directory,
        processed_directory,
        source,
        processed_store,
    )

    # 4. Raw unchanged
    change = _raw_size_change(raw_directory, raw_store, sizes)

    # 5. Publish or refuse
    split_names = tuple(item.split for item in staged)
    if stage_error is not None or change is not None:
        processed_store.discard(processed_directory, split_names)
        _raise_preparation_failure(stage_error, change)
    processed_store.publish(processed_directory, split_names)

    # 6. Preparation report
    return PrepareOutcome(
        raw_directory=raw_directory,
        processed_directory=processed_directory,
        splits=staged,
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


def _stage_splits(
    spec: DatasetSpec,
    raw_directory: Path,
    processed_directory: Path,
    source: RawStorySource,
    processed_store: ProcessedTextStore,
) -> tuple[tuple[PreparedSplit, ...], Exception | None]:
    """Stage every raw file, keeping a stage error instead of publishing.

    Args:
        spec: Dataset whose raw file names are prepared.
        raw_directory: Directory of immutable raw JSONL files.
        processed_directory: Directory for derived JSONL files.
        source: Reader for story text in one raw JSONL file.
        processed_store: Writer for staged split files.

    Returns:
        The staged splits completed before a failure, and that failure or None.

    Flow:
        1. Stage one split — append each raw file's staged result.
        2. Stage failure — return the splits already staged and the error.
    """
    staged: list[PreparedSplit] = []
    try:
        # 1. Stage one split
        staged.extend(
            _stage_split(
                raw_directory,
                processed_directory,
                filename,
                source,
                processed_store,
            )
            for filename in spec.filenames
        )
    except _STAGE_ERRORS as error:
        # 2. Stage failure
        return tuple(staged), error
    return tuple(staged), None


def _stage_split(
    raw_directory: Path,
    processed_directory: Path,
    filename: str,
    source: RawStorySource,
    processed_store: ProcessedTextStore,
) -> PreparedSplit:
    """Stream one raw file into that split's partial JSONL.

    Args:
        raw_directory: Directory of immutable raw JSONL files.
        processed_directory: Directory for derived JSONL files.
        filename: Raw file name to read.
        source: Reader for story text in one raw JSONL file.
        processed_store: Writer for staged split files.

    Returns:
        The split, story count, and derived file name.

    Flow:
        1. Split name — take the published split token from the raw file name.
        2. Story text — stream records from that raw file.
        3. Partial file — stage those stories and return the count.
    """
    # 1. Split name
    split = split_from_filename(filename)

    # 2. Story text
    texts = source.texts(raw_directory / filename)

    # 3. Partial file
    story_count = processed_store.stage(processed_directory, split, texts)
    return PreparedSplit(
        raw_filename=filename,
        split=split,
        story_count=story_count,
        output_name=processed_filename(split),
    )


def _raw_size_change(
    raw_directory: Path,
    raw_store: RawDatasetStore,
    sizes: dict[str, int],
) -> RawDatasetChangedError | None:
    """Return a size-change error when a raw file no longer has its recorded size.

    Args:
        raw_directory: Directory of immutable raw JSONL files.
        raw_store: Size checks for the raw files.
        sizes: Byte count by raw file name recorded before writing.

    Returns:
        The size-change error, or None when every recorded size still matches.
    """
    for filename, size in sizes.items():
        current = raw_store.byte_count(raw_directory, filename)
        if current != size:
            message = (
                f"Raw file {raw_directory / filename} changed from {size} "
                f"bytes to {current} bytes while preparing derived text."
            )
            return RawDatasetChangedError(message)
    return None


def _raise_preparation_failure(
    stage_error: Exception | None,
    change: RawDatasetChangedError | None,
) -> NoReturn:
    """Raise a stage failure, a raw size change, or the stage failure caused by it.

    Args:
        stage_error: Failure while staging a split, or None when staging finished.
        change: Raw size change, or None when every recorded size still matches.

    Raises:
        RawStoryRecordError: Staging failed on a story line and no size change
            was recorded, or a size change is attached as the cause.
        UnknownDatasetSplitError: Staging failed on the split token.
        OSError: Staging failed while writing a partial file.
        RawDatasetChangedError: Staging finished and a raw file's size changed.
        RuntimeError: Neither a stage error nor a size change was provided.
    """
    if stage_error is not None and change is not None:
        raise stage_error from change
    if stage_error is not None:
        raise stage_error
    if change is not None:
        raise change
    message = "Preparation failed without a stage error or a raw size change."
    raise RuntimeError(message)
