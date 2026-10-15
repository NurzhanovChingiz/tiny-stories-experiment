"""Wire dataset acquisition from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_story.application.use_cases.download_dataset import download_dataset
from tiny_story.infrastructure.config.dataset_config import (
    DatasetConfig,
    HuggingFaceSourceConfig,
    load_dataset_config,
)
from tiny_story.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)
from tiny_story.infrastructure.datasets.huggingface_dataset_source import (
    HuggingFaceDatasetSource,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tiny_story.application.results.download_outcome import DownloadOutcome


def acquire_dataset(
    config_path: Path,
    destination: Path | None = None,
    cache_dir: Path | None = None,
) -> DownloadOutcome:
    """Acquire the dataset described by a configuration file.

    Args:
        config_path: YAML dataset configuration.
        destination: Raw directory that replaces ``dataset.destination`` when set.
        cache_dir: Hugging Face cache that replaces ``dataset.source.cache_dir`` when set.

    Returns:
        Which raw files were stored and which were already complete.

    Flow:
        1. Load the dataset configuration.
        2. Apply caller overrides to the destination and cache.
        3. Download the configured Hugging Face files into raw storage.
    """
    # 1. Load the dataset configuration.
    config = load_dataset_config(config_path)

    # 2. Apply caller overrides to the destination and cache.
    config = _with_overrides(config, destination=destination, cache_dir=cache_dir)

    # 3. Download the configured Hugging Face files into raw storage.
    source = HuggingFaceDatasetSource(
        cache_dir=config.source.cache_dir,
        repository=config.source.repository,
        revision=config.source.revision,
    )
    store = FilesystemDatasetRepository(config.destination)
    return download_dataset(config.spec, source, store, existing=config.existing)


def _with_overrides(
    config: DatasetConfig,
    *,
    destination: Path | None,
    cache_dir: Path | None,
) -> DatasetConfig:
    """Return configuration with optional destination and cache overrides.

    Args:
        config: Parsed dataset configuration.
        destination: Replacement raw directory, or None to keep the configured one.
        cache_dir: Replacement Hugging Face cache, or None to keep the configured one.

    Returns:
        The configuration acquisition should use.
    """
    source = config.source
    if cache_dir is not None:
        source = HuggingFaceSourceConfig(
            repository=source.repository,
            revision=source.revision,
            cache_dir=cache_dir,
        )
    raw_dir = config.destination if destination is None else destination
    return DatasetConfig(
        spec=config.spec,
        destination=raw_dir,
        existing=config.existing,
        source=source,
    )
