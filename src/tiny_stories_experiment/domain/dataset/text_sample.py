"""One story from a published TinyStories split."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from tiny_stories_experiment.domain.errors import UnknownDatasetSplitError


class DatasetSplit(StrEnum):
    """Published TinyStoriesV2 split named in the raw file.

    Members:
        TRAIN: Stories from ``TinyStoriesV2-GPT4-train.jsonl``.
        VALID: Stories from ``TinyStoriesV2-GPT4-valid.jsonl``.
    """

    TRAIN = "train"
    VALID = "valid"


@dataclass(frozen=True)
class TextSample:
    """Story text taken from one published split.

    Attributes:
        text: Story text from one published record.
        split: Split that record was read from.
    """

    text: str
    split: DatasetSplit


def split_from_filename(filename: str) -> DatasetSplit:
    """Return the published split named by a raw dataset file.

    The split is the last hyphen-separated token of the file stem, so
    ``TinyStoriesV2-GPT4-train.jsonl`` names the train split.

    Args:
        filename: Raw dataset file name, with or without a directory prefix.

    Returns:
        The train or valid split named by that file.

    Raises:
        UnknownDatasetSplitError: The file stem does not end in ``train`` or
            ``valid``.

    Flow:
        1. Split token — take the last hyphen-separated stem segment.
        2. Known split — return that published split, or refuse another token.
    """
    # 1. Split token
    label = PurePosixPath(filename).stem.rsplit("-", maxsplit=1)[-1]

    # 2. Known split
    try:
        return DatasetSplit(label)
    except ValueError as error:
        message = f"Raw file {filename} does not name a train or valid split."
        raise UnknownDatasetSplitError(message) from error
