"""Use-case tests for training a tokenizer on prepared split files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from tokenizers import Tokenizer

from tiny_stories_experiment.application.ports.raw_story_source import RawStorySource
from tiny_stories_experiment.application.ports.tokenizer_repository import (
    TOKENIZER_FILENAME,
)
from tiny_stories_experiment.application.use_cases.train_tokenizer import (
    train_tokenizer,
)
from tiny_stories_experiment.domain.errors import (
    ProcessedTextChangedError,
    ProcessedTextMissingError,
    RawStoryRecordError,
    TokenizerTrainingError,
)
from tiny_stories_experiment.domain.tokenization.tokenizer_spec import TokenizerSpec
from tiny_stories_experiment.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)
from tiny_stories_experiment.infrastructure.datasets.jsonl_story_source import (
    JsonlStorySource,
)
from tiny_stories_experiment.infrastructure.tokenization.bpe_tokenizer_trainer import (
    BpeTokenizerTrainer,
)
from tiny_stories_experiment.infrastructure.tokenization.filesystem_tokenizer_repository import (
    FilesystemTokenizerRepository,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

STORIES = (
    "alpha alpha alpha.",
    "alpha beta alpha.",
    "beta beta beta.",
)
SPEC = TokenizerSpec(
    name="fixture_bpe",
    vocab_size=300,
    min_frequency=2,
    special_tokens=("<|endoftext|>",),
)


def _write_split(directory: Path, filename: str, stories: tuple[str, ...]) -> None:
    """Write story records as JSONL.

    Args:
        directory: Directory that receives the file.
        filename: JSONL file name.
        stories: Story text values, in order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"text": story}) for story in stories]
    text = "\n".join(lines)
    if text:
        text = f"{text}\n"
    (directory / filename).write_text(text, encoding="utf-8")


def _write_fixture(processed: Path, stories: tuple[str, ...] = STORIES) -> None:
    """Write the same stories into both prepared splits.

    Args:
        processed: Directory for ``train.jsonl`` and ``valid.jsonl``.
        stories: Story text values, in order.
    """
    _write_split(processed, "train.jsonl", stories)
    _write_split(processed, "valid.jsonl", stories)


def _bytes_by_name(directory: Path) -> dict[str, bytes]:
    """Read both prepared split files.

    Args:
        directory: Directory that contains the split files.

    Returns:
        File contents keyed by name.
    """
    return {
        name: (directory / name).read_bytes() for name in ("train.jsonl", "valid.jsonl")
    }


def _train(processed: Path, artifacts: Path, spec: TokenizerSpec = SPEC) -> Path:
    """Train ``spec`` from ``processed`` into ``artifacts``.

    Args:
        processed: Directory of prepared JSONL files.
        artifacts: Directory for the tokenizer document.
        spec: Tokenizer limits.

    Returns:
        The stored tokenizer document.
    """
    outcome = train_tokenizer(
        spec,
        processed,
        artifacts,
        source=JsonlStorySource(),
        trainer=BpeTokenizerTrainer(),
        repository=FilesystemTokenizerRepository(),
        file_sizes=FilesystemDatasetRepository(),
    )
    return outcome.artifact_path


def test_train_tokenizer_learns_fixture_merges_without_changing_sources(
    tmp_path: Path,
) -> None:
    """A tiny prepared corpus yields a byte-level tokenizer and identical JSONL."""
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    _write_fixture(processed)
    before = _bytes_by_name(processed)

    artifact = _train(processed, artifacts)
    tokenizer = Tokenizer.from_file(str(artifact))

    assert artifact == artifacts / SPEC.name / TOKENIZER_FILENAME
    assert not artifact.with_name(f"{TOKENIZER_FILENAME}.partial").exists()
    assert _bytes_by_name(processed) == before
    assert tokenizer.get_vocab_size() <= SPEC.vocab_size
    assert len(tokenizer.encode("alpha").ids) == 1
    for text in (*STORIES, "", "zzz"):
        assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_train_tokenizer_refuses_a_missing_split(tmp_path: Path) -> None:
    """Training does not start when valid.jsonl is absent."""
    processed = tmp_path / "processed"
    _write_split(processed, "train.jsonl", STORIES)

    with pytest.raises(ProcessedTextMissingError, match=r"valid\.jsonl"):
        _train(processed, tmp_path / "artifacts")

    assert not (tmp_path / "artifacts").exists()


def test_train_tokenizer_refuses_an_empty_corpus(tmp_path: Path) -> None:
    """Empty split files do not produce a tokenizer document."""
    processed = tmp_path / "processed"
    _write_fixture(processed, ())
    before = _bytes_by_name(processed)

    with pytest.raises(TokenizerTrainingError, match="no stories"):
        _train(processed, tmp_path / "artifacts")

    assert _bytes_by_name(processed) == before
    assert not (tmp_path / "artifacts").exists()


def test_train_tokenizer_refuses_a_vocab_below_the_byte_alphabet(
    tmp_path: Path,
) -> None:
    """A vocab smaller than the byte alphabet is not stored."""
    processed = tmp_path / "processed"
    _write_fixture(processed)
    before = _bytes_by_name(processed)
    spec = TokenizerSpec("tiny", vocab_size=10, min_frequency=1, special_tokens=())

    with pytest.raises(TokenizerTrainingError, match="byte-level minimum"):
        _train(processed, tmp_path / "artifacts", spec)

    assert _bytes_by_name(processed) == before
    assert not (tmp_path / "artifacts").exists()


def test_train_tokenizer_leaves_sources_when_a_story_line_is_invalid(
    tmp_path: Path,
) -> None:
    """A bad JSONL line does not publish a tokenizer or rewrite the splits."""
    processed = tmp_path / "processed"
    _write_fixture(processed)
    train = processed / "train.jsonl"
    train.write_text(train.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")
    before = _bytes_by_name(processed)

    with pytest.raises(RawStoryRecordError, match="not a story record"):
        _train(processed, tmp_path / "artifacts")

    assert _bytes_by_name(processed) == before
    assert not (tmp_path / "artifacts").exists()


class _ExpandingSource(RawStorySource):
    """Append one byte to each prepared file after reading it."""

    def __init__(self) -> None:
        """Read stories with the JSONL adapter."""
        self._inner = JsonlStorySource()

    def texts(self, path: Path) -> Iterator[str]:
        """Yield the file's stories, then change that file's size.

        Args:
            path: Prepared JSONL file to read.

        Returns:
            Story text in file order.
        """
        yield from self._inner.texts(path)
        path.write_bytes(path.read_bytes() + b" ")


def test_train_tokenizer_does_not_publish_when_a_split_changes_size(
    tmp_path: Path,
) -> None:
    """A prepared file whose size changes during training is not published."""
    processed = tmp_path / "processed"
    _write_fixture(processed)

    with pytest.raises(ProcessedTextChangedError, match="changed from"):
        train_tokenizer(
            SPEC,
            processed,
            tmp_path / "artifacts",
            source=_ExpandingSource(),
            trainer=BpeTokenizerTrainer(),
            repository=FilesystemTokenizerRepository(),
            file_sizes=FilesystemDatasetRepository(),
        )

    assert not (tmp_path / "artifacts").exists()
