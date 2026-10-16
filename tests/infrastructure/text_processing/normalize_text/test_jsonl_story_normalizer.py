"""Tests for derived JSONL story normalization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tiny_story.domain.errors import NormalizeTextError
from tiny_story.infrastructure.text_processing.normalize_text.jsonl_story_normalizer import (
    JsonlStoryNormalizer,
)


def _normalizer() -> JsonlStoryNormalizer:
    """Return a normalizer that inserts the sentence delimiter.

    Returns:
        A story-file normalizer.
    """
    return JsonlStoryNormalizer(enabled=True, sentence_delimiter="|")


def test_normalize_file_marks_sentences_and_leaves_the_source(tmp_path: Path) -> None:
    """A derived file holds delimited sentences and the source file stays unchanged."""
    source = tmp_path / "interim" / "valid.jsonl"
    source.parent.mkdir()
    original = '{"text": "hello world.", "story_id": 7}\n'
    source.write_text(original, encoding="utf-8")
    destination = tmp_path / "processed" / "valid.jsonl"

    _normalizer().normalize_file(source, destination)

    assert source.read_text(encoding="utf-8") == original
    story = json.loads(destination.read_text(encoding="utf-8"))
    assert story["story_id"] == 7
    assert story["text"].endswith(".|")


def test_normalize_file_rejects_a_missing_source(tmp_path: Path) -> None:
    """A missing story file is not treated as an empty derived file."""
    source = tmp_path / "missing.jsonl"

    with pytest.raises(NormalizeTextError, match="does not exist"):
        _normalizer().normalize_file(source, tmp_path / "processed.jsonl")


def test_normalize_file_refuses_to_replace_the_raw_file(tmp_path: Path) -> None:
    """The normalizer does not write the derived text over the raw file."""
    source = tmp_path / "valid.jsonl"
    source.write_text('{"text": "hello."}\n', encoding="utf-8")

    with pytest.raises(NormalizeTextError, match="Refusing to replace"):
        _normalizer().normalize_file(source, source)

    assert source.read_text(encoding="utf-8") == '{"text": "hello."}\n'
