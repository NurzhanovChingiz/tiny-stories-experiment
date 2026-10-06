"""Identity tests for a story and the split it came from."""

from __future__ import annotations

import pytest

from tiny_stories_experiment.domain.dataset.text_sample import (
    DatasetSplit,
    TextSample,
    split_from_filename,
)
from tiny_stories_experiment.domain.errors import UnknownDatasetSplitError


def test_text_sample_keeps_story_and_split() -> None:
    """A sample retains the story text and the split it was taken from."""
    sample = TextSample(text="Once upon a time.", split=DatasetSplit.TRAIN)

    assert sample.text == "Once upon a time."
    assert sample.split is DatasetSplit.TRAIN


def test_same_story_on_another_split_is_a_different_sample() -> None:
    """Train and validation copies of the same story stay distinct."""
    story = "Once upon a time."
    train = TextSample(text=story, split=DatasetSplit.TRAIN)
    valid = TextSample(text=story, split=DatasetSplit.VALID)

    assert train != valid


def test_split_names_match_published_filenames() -> None:
    """Split values are the tokens used in the raw JSONL file names."""
    assert DatasetSplit.TRAIN.value == "train"
    assert DatasetSplit.VALID.value == "valid"
    assert DatasetSplit("train") is DatasetSplit.TRAIN
    assert DatasetSplit("valid") is DatasetSplit.VALID


def test_unknown_split_name_is_rejected() -> None:
    """A name outside the published train and valid files is not a split."""
    with pytest.raises(ValueError, match="nope"):
        DatasetSplit("nope")


def test_published_filename_names_its_split() -> None:
    """The last token of a raw file name is the split that file belongs to."""
    assert split_from_filename("TinyStoriesV2-GPT4-train.jsonl") is DatasetSplit.TRAIN
    assert split_from_filename("TinyStoriesV2-GPT4-valid.jsonl") is DatasetSplit.VALID


def test_filename_without_a_published_split_is_rejected() -> None:
    """A raw file whose stem does not end in train or valid has no split."""
    with pytest.raises(UnknownDatasetSplitError, match=r"notes\.jsonl"):
        split_from_filename("notes.jsonl")
