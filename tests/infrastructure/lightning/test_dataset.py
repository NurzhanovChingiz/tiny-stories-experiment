"""Tests for tokenized story loading and windowing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.infrastructure.lightning.corpus import write_tokenized_corpus
from tiny_story.domain.errors import TrainError

try:
    import torch

    from tiny_story.infrastructure.lightning.dataset import (
        TokenizedStoryDataset,
        load_token_id_sequences,
        load_vocabulary_size,
        window_sequences,
    )
except ImportError:
    pytest.skip("torch is required", allow_module_level=True)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_token_id_sequences_reads_jsonl_stories(tmp_path: Path) -> None:
    """Token ids are collected from every JSONL story."""
    write_tokenized_corpus(tmp_path, stories=2)

    sequences = load_token_id_sequences(tmp_path)

    assert sequences == ((2, 4, 5, 4, 5, 3), (2, 4, 5, 4, 5, 3))
    assert load_vocabulary_size(tmp_path / "vocab.json") == 6


def test_window_sequences_pads_and_truncates() -> None:
    """Short sequences pad; long sequences keep the prefix."""
    windows = window_sequences(((1, 2), (1, 2, 3, 4, 5)), sequence_length=4)

    assert windows == ((1, 2, 0, 0), (1, 2, 3, 4))


def test_load_token_id_sequences_rejects_a_line_without_ids(tmp_path: Path) -> None:
    """A story object without token ids is rejected."""
    (tmp_path / "train.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")

    with pytest.raises(TrainError, match="token_ids"):
        load_token_id_sequences(tmp_path)


def test_load_vocabulary_size_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing vocabulary file is rejected."""
    with pytest.raises(TrainError, match="does not exist"):
        load_vocabulary_size(tmp_path / "vocab.json")


def test_tokenized_story_dataset_returns_long_tensors() -> None:
    """Each window is a long tensor of the stored ids."""
    dataset = TokenizedStoryDataset(((2, 4, 5, 3),))

    item = dataset[0]

    assert len(dataset) == 1
    assert item.dtype == torch.long
    assert item.tolist() == [2, 4, 5, 3]


def test_load_token_id_sequences_rejects_invalid_json(tmp_path: Path) -> None:
    """A broken JSONL line is rejected."""
    (tmp_path / "train.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(TrainError, match="Could not read tokenized file"):
        load_token_id_sequences(tmp_path)
