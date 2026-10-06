"""Wire TinyStoriesV2 download and preparation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tiny_stories_experiment.application.use_cases.download_dataset import (
    download_dataset,
)
from tiny_stories_experiment.application.use_cases.prepare_dataset import (
    prepare_dataset,
)
from tiny_stories_experiment.domain.dataset.dataset_spec import DatasetSpec
from tiny_stories_experiment.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)
from tiny_stories_experiment.infrastructure.datasets.filesystem_processed_text_store import (
    FilesystemProcessedTextStore,
)
from tiny_stories_experiment.infrastructure.datasets.huggingface_tinystories_repository import (
    HuggingfaceTinystoriesRepository,
)
from tiny_stories_experiment.infrastructure.datasets.jsonl_story_source import (
    JsonlStorySource,
)

if TYPE_CHECKING:
    from tiny_stories_experiment.application.results.download_outcome import (
        DownloadOutcome,
    )
    from tiny_stories_experiment.application.results.prepare_outcome import (
        PrepareOutcome,
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
DEFAULT_PROCESSED_DESTINATION = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "tiny_stories"
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


def prepare_default_dataset(
    raw_directory: Path = DEFAULT_RAW_DESTINATION,
    processed_directory: Path = DEFAULT_PROCESSED_DESTINATION,
) -> PrepareOutcome:
    """Prepare TinyStoriesV2 raw JSONL into derived split files.

    Args:
        raw_directory: Directory of immutable raw JSONL files.
        processed_directory: Directory for derived ``train.jsonl`` and
            ``valid.jsonl`` files.

    Returns:
        The split, story count, and derived file name for each raw file.
    """
    return prepare_dataset(
        TINY_STORIES_V2,
        raw_directory,
        processed_directory,
        source=JsonlStorySource(),
        processed_store=FilesystemProcessedTextStore(),
        raw_store=FilesystemDatasetRepository(),
    )
