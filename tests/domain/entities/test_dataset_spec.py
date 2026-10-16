"""Behavior of dataset identity checks."""

from __future__ import annotations

import pytest

from tiny_story.domain.entities.dataset_spec import (
    DatasetSpec,
    require_dataset_spec,
    require_raw_filename,
)
from tiny_story.domain.errors import InvalidDatasetSpecError, InvalidRawFilenameError


def test_raw_filename_accepts_a_single_file_name() -> None:
    """A flat file name can be joined onto the raw directory."""
    assert require_raw_filename("TinyStoriesV2-GPT4-train.jsonl") == (
        "TinyStoriesV2-GPT4-train.jsonl"
    )


@pytest.mark.parametrize(
    "filename",
    ["", " ", ".", "..", "a/b.jsonl", "a\\b.jsonl", "/abs/a.jsonl", "a\x00b"],
)
def test_raw_filename_rejects_a_name_that_can_leave_the_directory(
    filename: str,
) -> None:
    """Empty, absolute, and path-shaped names are rejected."""
    with pytest.raises(InvalidRawFilenameError, match="single file name"):
        require_raw_filename(filename)


def test_dataset_spec_accepts_distinct_raw_filenames() -> None:
    """A name and two flat filenames can be stored."""
    spec = DatasetSpec(name="tiny_stories", filenames=("train.jsonl", "valid.jsonl"))

    assert require_dataset_spec(spec) is spec


@pytest.mark.parametrize(
    "filenames",
    [(), ("train.jsonl", "train.jsonl")],
)
def test_dataset_spec_rejects_missing_or_repeated_files(
    filenames: tuple[str, ...],
) -> None:
    """Each raw file must be named once."""
    spec = DatasetSpec(name="corpus", filenames=filenames)

    with pytest.raises(InvalidDatasetSpecError, match="name each raw file once"):
        require_dataset_spec(spec)


def test_dataset_spec_rejects_a_blank_name() -> None:
    """A blank dataset name cannot be stored."""
    spec = DatasetSpec(name="  ", filenames=("train.jsonl",))

    with pytest.raises(InvalidDatasetSpecError, match="Dataset name"):
        require_dataset_spec(spec)
