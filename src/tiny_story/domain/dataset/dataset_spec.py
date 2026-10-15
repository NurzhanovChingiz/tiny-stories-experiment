"""Identity of a published text dataset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Identity of a published text dataset.

    Attributes:
        repo_id: Hugging Face dataset repository id.
        source_url: Page that documents the dataset.
        filenames: Published file names to place in raw storage.
    """

    repo_id: str
    source_url: str
    filenames: tuple[str, ...]
