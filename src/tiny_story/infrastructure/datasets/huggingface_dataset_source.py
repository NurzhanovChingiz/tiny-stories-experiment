"""Hugging Face adapter for one configured dataset."""

from __future__ import annotations

from pathlib import Path

import httpx2
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    XetDownloadError,
)

from tiny_story.application.ports.dataset_source import DatasetSource
from tiny_story.domain.errors import DownloadError


class HuggingFaceDatasetSource(DatasetSource):
    """Downloads one file from a configured Hugging Face dataset repository."""

    def __init__(self, cache_dir: Path, repository: str, revision: str) -> None:
        """Store the repository, revision, and interim cache.

        Args:
            cache_dir: Directory for the Hugging Face cache.
            repository: Hugging Face dataset id, such as ``owner/name``.
            revision: Dataset revision, branch, tag, or commit.

        Returns:
            None.
        """
        self._cache_dir = cache_dir
        self._repository = repository
        self._revision = revision

    def fetch(self, filename: str) -> Path:
        """Download one dataset file and return its cache path.

        Args:
            filename: File name inside the configured repository.

        Returns:
            Path of the downloaded file. The file is not modified.

        Raises:
            DownloadError: The file could not be retrieved.

        Flow:
            1. Download the requested file into the adapter cache.
            2. Return the cache path when the download produced a file.
        """
        # 1. Download the requested file into the adapter cache.
        try:
            downloaded = hf_hub_download(
                repo_id=self._repository,
                filename=filename,
                repo_type="dataset",
                revision=self._revision,
                cache_dir=self._cache_dir,
            )
        except (
            EntryNotFoundError,
            GatedRepoError,
            HFValidationError,
            HfHubHTTPError,
            LocalEntryNotFoundError,
            OSError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
            XetDownloadError,
            httpx2.TransportError,
        ) as exc:
            message = f"Could not download {filename} from {self._repository}"
            raise DownloadError(message) from exc

        # 2. Return the cache path when the download produced a file.
        if not isinstance(downloaded, str):
            message = f"Could not download {filename} from {self._repository}"
            raise DownloadError(message)
        return Path(downloaded)
