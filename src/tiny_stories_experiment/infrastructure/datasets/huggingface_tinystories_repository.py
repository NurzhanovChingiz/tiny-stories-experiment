"""Hugging Face Hub adapter for a published TinyStories dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

from tiny_stories_experiment.application.ports.published_dataset_source import (
    PublishedDatasetSource,
)
from tiny_stories_experiment.domain.errors import PublishedSizeMissingError

if TYPE_CHECKING:
    from pathlib import Path


class HuggingfaceTinystoriesRepository(PublishedDatasetSource):
    """Hugging Face Hub source for one dataset repository."""

    def __init__(self, repo_id: str) -> None:
        """Bind the repository this source reads.

        Args:
            repo_id: Hugging Face dataset repository id.
        """
        self._repo_id = repo_id

    def byte_count(self, filename: str) -> int:
        """Return the Hub byte count for one dataset file.

        Args:
            filename: File name inside the dataset repository.

        Returns:
            The published file size in bytes.

        Raises:
            PublishedSizeMissingError: Hub metadata does not include a file size.

        Flow:
            1. Hub metadata — request the published file record.
            2. Required size — return the byte count when the record includes one.
        """
        # 1. Hub metadata
        url = hf_hub_url(
            repo_id=self._repo_id,
            filename=filename,
            repo_type="dataset",
        )
        metadata = get_hf_file_metadata(url)

        # 2. Required size
        size = metadata.size
        if size is None:
            message = f"Hub metadata for {filename} has no size"
            raise PublishedSizeMissingError(message)
        return size

    def fetch(self, filename: str, destination: Path) -> None:
        """Download one dataset file into a local directory.

        Args:
            filename: File name inside the dataset repository.
            destination: Directory that receives the file.
        """
        hf_hub_download(
            repo_id=self._repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=destination,
        )
