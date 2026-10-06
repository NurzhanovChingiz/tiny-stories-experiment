"""Wire the TinyStoriesV2 raw download."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tiny_stories_experiment.application.use_cases.download_dataset import (
    download_dataset,
)
from tiny_stories_experiment.domain.dataset.dataset_spec import DatasetSpec
from tiny_stories_experiment.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)
from tiny_stories_experiment.infrastructure.datasets.huggingface_tinystories_repository import (
    HuggingfaceTinystoriesRepository,
)

if TYPE_CHECKING:
    from tiny_stories_experiment.application.results.download_outcome import (
        DownloadOutcome,
    )

TINY_STORIES_V2 = DatasetSpec(
    repo_id="noanabeshima/TinyStoriesV2",
    source_url="https://huggingface.co/datasets/noanabeshima/TinyStoriesV2",
    filenames=(
        "TinyStoriesV2-GPT4-train.jsonl",
        "TinyStoriesV2-GPT4-valid.jsonl",
    ),
)
DEFAULT_RAW_DESTINATION = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "tiny_stories_raw"
)


def download_default_dataset(
    destination: Path = DEFAULT_RAW_DESTINATION,
) -> DownloadOutcome:
    """Download TinyStoriesV2 into the raw directory.

    Args:
        destination: Directory for the immutable raw files.

    Returns:
        Which files were already present and which were fetched.
    """
    return download_dataset(
        spec=TINY_STORIES_V2,
        destination=destination,
        source=HuggingfaceTinystoriesRepository(TINY_STORIES_V2.repo_id),
        store=FilesystemDatasetRepository(),
    )
