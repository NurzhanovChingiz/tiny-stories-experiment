"""Train a tokenizer on prepared splits without changing those files."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from tiny_stories_experiment.application.ports.processed_text_store import (
    processed_filename,
)
from tiny_stories_experiment.application.results.train_tokenizer_outcome import (
    TrainTokenizerOutcome,
)
from tiny_stories_experiment.domain.dataset.text_sample import DatasetSplit
from tiny_stories_experiment.domain.errors import (
    ProcessedTextChangedError,
    ProcessedTextMissingError,
    RawStoryRecordError,
    TokenizerTrainingError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tiny_stories_experiment.application.ports.directory_file_size import (
        DirectoryFileSize,
    )
    from tiny_stories_experiment.application.ports.raw_story_source import (
        RawStorySource,
    )
    from tiny_stories_experiment.application.ports.tokenizer_repository import (
        TokenizerRepository,
    )
    from tiny_stories_experiment.application.ports.tokenizer_trainer import (
        TokenizerTrainer,
    )
    from tiny_stories_experiment.application.results.trained_tokenizer import (
        TrainedTokenizer,
    )
    from tiny_stories_experiment.domain.tokenization.tokenizer_spec import (
        TokenizerSpec,
    )

_PREPARED_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VALID)
_TRAIN_ERRORS = (OSError, RawStoryRecordError, TokenizerTrainingError)


def train_tokenizer(
    spec: TokenizerSpec,
    processed_directory: Path,
    artifact_directory: Path,
    *,
    source: RawStorySource,
    trainer: TokenizerTrainer,
    repository: TokenizerRepository,
    file_sizes: DirectoryFileSize,
) -> TrainTokenizerOutcome:
    """Train on ``train.jsonl`` and ``valid.jsonl`` and store the tokenizer.

    The tokenizer document is written only after training finishes and both
    prepared files still have the byte sizes recorded at the start. Those
    files are read and never written by this use case.

    Args:
        spec: Tokenizer name, vocab size, minimum frequency, and special tokens.
        processed_directory: Directory of derived ``train.jsonl`` and
            ``valid.jsonl`` files.
        artifact_directory: Directory that will hold ``spec.name/tokenizer.json``.
        source: Reader for story text in one prepared JSONL file.
        trainer: BPE trainer for the story stream.
        repository: Writer for the serialized tokenizer.
        file_sizes: Size checks for the prepared files.

    Returns:
        The artifact path, trained vocab size, and story count.

    Raises:
        ProcessedTextMissingError: ``train.jsonl`` or ``valid.jsonl`` is absent.
        RawStoryRecordError: A prepared line is not an object with a string
            text field. A size change during that failure is attached as the
            cause.
        TokenizerTrainingError: The trainer cannot build a vocabulary from
            the spec, or the splits contain no stories. A size change during
            that failure is attached as the cause.
        ProcessedTextChangedError: Training finished and a prepared file's
            size changed.
        OSError: Reading a prepared file or storing the tokenizer failed.

    Flow:
        1. Source sizes — record each prepared split's byte count.
        2. Train — stream both splits into the trainer.
        3. Sizes unchanged — read the recorded sizes again.
        4. Publish or refuse — store the document, or raise without storing it.
        5. Training report — return the artifact path, vocab size, and story count.
    """
    # 1. Source sizes
    sizes = _processed_sizes(processed_directory, file_sizes)

    # 2. Train
    trained, train_error = _train(spec, processed_directory, source, trainer)

    # 3. Sizes unchanged
    change = _size_change(processed_directory, file_sizes, sizes)

    # 4. Publish or refuse
    if trained is None or train_error is not None or change is not None:
        _raise_training_failure(train_error, change)
    artifact_path = repository.save(artifact_directory, spec.name, trained.content)

    # 5. Training report
    return TrainTokenizerOutcome(
        processed_directory=processed_directory,
        artifact_path=artifact_path,
        vocab_size=trained.vocab_size,
        story_count=trained.story_count,
    )


def _processed_sizes(
    processed_directory: Path,
    file_sizes: DirectoryFileSize,
) -> dict[str, int]:
    """Record the byte count of ``train.jsonl`` and ``valid.jsonl``.

    Args:
        processed_directory: Directory of derived split files.
        file_sizes: Size checks for the prepared files.

    Returns:
        Byte count by prepared file name.

    Raises:
        ProcessedTextMissingError: A prepared split file is absent.

    Flow:
        1. Local size — read the byte count of each prepared split.
        2. Missing file — refuse when a split file is absent.
    """
    sizes: dict[str, int] = {}
    for split in _PREPARED_SPLITS:
        filename = processed_filename(split)
        # 1. Local size
        size = file_sizes.byte_count(processed_directory, filename)
        # 2. Missing file
        if size is None:
            message = f"Prepared file {processed_directory / filename} is absent."
            raise ProcessedTextMissingError(message)
        sizes[filename] = size
    return sizes


def _train(
    spec: TokenizerSpec,
    processed_directory: Path,
    source: RawStorySource,
    trainer: TokenizerTrainer,
) -> tuple[TrainedTokenizer | None, Exception | None]:
    """Train on both prepared splits, keeping a training error instead of saving.

    Args:
        spec: Tokenizer name, vocab size, minimum frequency, and special tokens.
        processed_directory: Directory of derived split files.
        source: Reader for story text in one prepared JSONL file.
        trainer: BPE trainer for the story stream.

    Returns:
        The trained tokenizer, or None with the training error.

    Flow:
        1. Story stream — train on train.jsonl, then valid.jsonl.
        2. Training failure — return the error instead of a tokenizer.
        3. Empty corpus — refuse a run that read no stories.
    """
    try:
        # 1. Story stream
        trained = trainer.train(spec, _stories(processed_directory, source))
    except _TRAIN_ERRORS as error:
        # 2. Training failure
        return None, error
    # 3. Empty corpus
    if trained.story_count == 0:
        message = "Prepared splits contain no stories."
        return None, TokenizerTrainingError(message)
    return trained, None


def _stories(processed_directory: Path, source: RawStorySource) -> Iterator[str]:
    """Yield stories from ``train.jsonl`` and then ``valid.jsonl``.

    Args:
        processed_directory: Directory of derived split files.
        source: Reader for story text in one prepared JSONL file.

    Returns:
        Story text in train-split order, followed by valid-split order.
    """
    for split in _PREPARED_SPLITS:
        yield from source.texts(processed_directory / processed_filename(split))


def _size_change(
    processed_directory: Path,
    file_sizes: DirectoryFileSize,
    sizes: dict[str, int],
) -> ProcessedTextChangedError | None:
    """Return a size-change error when a prepared file no longer matches.

    Args:
        processed_directory: Directory of derived split files.
        file_sizes: Size checks for the prepared files.
        sizes: Byte count by prepared file name recorded before training.

    Returns:
        The size-change error, or None when every recorded size still matches.
    """
    for filename, size in sizes.items():
        current = file_sizes.byte_count(processed_directory, filename)
        if current != size:
            message = (
                f"Prepared file {processed_directory / filename} changed from "
                f"{size} bytes to {current} bytes while training the tokenizer."
            )
            return ProcessedTextChangedError(message)
    return None


def _raise_training_failure(
    train_error: Exception | None,
    change: ProcessedTextChangedError | None,
) -> NoReturn:
    """Raise a training failure, a size change, or the training failure caused by it.

    Args:
        train_error: Failure while training, or None when training finished.
        change: Prepared size change, or None when every recorded size still matches.

    Raises:
        RawStoryRecordError: Training failed on a story line.
        TokenizerTrainingError: Training failed on the vocabulary or an empty corpus.
        OSError: Training failed while reading a prepared file.
        ProcessedTextChangedError: Training finished and a prepared file's size changed.
        RuntimeError: Neither a training error nor a size change was provided.
    """
    if train_error is not None and change is not None:
        raise train_error from change
    if train_error is not None:
        raise train_error
    if change is not None:
        raise change
    message = "Tokenizer training failed without a training error or a size change."
    raise RuntimeError(message)
