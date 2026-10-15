"""Configuration for Hugging Face dataset acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tiny_story.application.use_cases.download_dataset import ExistingFilePolicy
from tiny_story.domain.dataset.dataset_spec import DatasetSpec, require_dataset_spec
from tiny_story.domain.errors import (
    InvalidDatasetConfigError,
    InvalidDatasetSpecError,
    InvalidRawFilenameError,
)
from tiny_story.infrastructure.config.yaml_config_repository import YamlConfigRepository


@dataclass(frozen=True, slots=True)
class HuggingFaceSourceConfig:
    """Hugging Face Hub location for one dataset.

    Attributes:
        repository: Hub dataset id, ``owner/name``.
        revision: Git revision or branch.
        cache_dir: Local Hugging Face cache directory.
    """

    repository: str
    revision: str
    cache_dir: Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Validated settings for acquiring one dataset.

    Attributes:
        spec: Dataset identity and raw file names.
        destination: Directory that receives the raw files.
        existing: Whether a complete raw file is kept or replaced.
        source: Hugging Face repository, revision, and cache.
    """

    spec: DatasetSpec
    destination: Path
    existing: ExistingFilePolicy
    source: HuggingFaceSourceConfig


def _text(value: object, field: str) -> str:
    """Return a non-empty string that has no surrounding whitespace.

    Args:
        value: Candidate field value.
        field: Configuration path named in the error.

    Returns:
        The same string when it is non-empty and unpadded.

    Raises:
        InvalidDatasetConfigError: The value is missing, blank, or padded.
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        message = f"{field} must be a non-empty string"
        raise InvalidDatasetConfigError(message)
    return value


def _mapping(value: object, field: str) -> dict[str, Any]:
    """Return a mapping whose keys are strings.

    Args:
        value: Candidate field value.
        field: Configuration path named in the error.

    Returns:
        The same mapping when every key is a string.

    Raises:
        InvalidDatasetConfigError: The value is not a string-keyed mapping.
    """
    if not isinstance(value, dict):
        message = f"{field} must be a mapping"
        raise InvalidDatasetConfigError(message)

    if not all(isinstance(key, str) for key in value):
        message = f"{field} must use string keys"
        raise InvalidDatasetConfigError(message)

    return value


def parse_dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    """Validate and convert a YAML mapping into dataset configuration.

    Args:
        raw: Parsed YAML document.

    Returns:
        Dataset identity, destination, existing-file policy, and Hub source.

    Raises:
        InvalidDatasetConfigError: A required field is missing or invalid.

    Flow:
        1. Read the dataset and source mappings.
        2. Require a non-empty list of file names.
        3. Build a storable dataset spec.
        4. Require a boolean overwrite flag.
        5. Require an owner/name repository id and assemble the configuration.
    """
    # 1. Read the dataset and source mappings.
    dataset = _mapping(raw.get("dataset"), "dataset")
    source = _mapping(dataset.get("source"), "dataset.source")

    # 2. Require a non-empty list of file names.
    files = dataset.get("files")
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(name, str) for name in files)
    ):
        message = "dataset.files must list each raw file name once"
        raise InvalidDatasetConfigError(message)

    # 3. Build a storable dataset spec.
    try:
        spec = require_dataset_spec(
            DatasetSpec(
                name=_text(dataset.get("name"), "dataset.name"),
                filenames=tuple(files),
            )
        )
    except (InvalidDatasetSpecError, InvalidRawFilenameError) as exc:
        message = str(exc)
        raise InvalidDatasetConfigError(message) from exc

    # 4. Require a boolean overwrite flag.
    overwrite = dataset.get("overwrite", False)
    if not isinstance(overwrite, bool):
        message = "dataset.overwrite must be true or false"
        raise InvalidDatasetConfigError(message)

    # 5. Require an owner/name repository id and assemble the configuration.
    repository = _text(source.get("repository"), "dataset.source.repository")
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        message = (
            "dataset.source.repository must be a Hugging Face "
            "dataset id such as owner/name"
        )
        raise InvalidDatasetConfigError(message)

    return DatasetConfig(
        spec=spec,
        destination=Path(_text(dataset.get("destination"), "dataset.destination")),
        existing=(ExistingFilePolicy.REPLACE if overwrite else ExistingFilePolicy.KEEP),
        source=HuggingFaceSourceConfig(
            repository=repository,
            revision=_text(
                source.get("revision", "main"),
                "dataset.source.revision",
            ),
            cache_dir=Path(
                _text(
                    source.get("cache_dir", "data/interim/huggingface"),
                    "dataset.source.cache_dir",
                )
            ),
        ),
    )


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset configuration file.

    Args:
        path: YAML dataset configuration.

    Returns:
        The parsed dataset configuration.

    Raises:
        InvalidDatasetConfigError: The file is missing, unreadable, or invalid.

    Flow:
        1. Reject a missing configuration file.
        2. Load the mapping and report read, YAML, and shape failures.
    """
    # 1. Reject a missing configuration file.
    if not path.is_file():
        message = f"Dataset configuration file does not exist: {path}"
        raise InvalidDatasetConfigError(message)

    # 2. Load the mapping and report read, YAML, and shape failures.
    repository = YamlConfigRepository(
        {DatasetConfig: path},
        {DatasetConfig: parse_dataset_config},
    )

    try:
        return repository.get(DatasetConfig)
    except OSError as exc:
        message = f"Could not read dataset configuration: {path}"
        raise InvalidDatasetConfigError(message) from exc
    except yaml.YAMLError as exc:
        message = f"Dataset configuration is not valid YAML: {path}"
        raise InvalidDatasetConfigError(message) from exc
    except TypeError as exc:
        message = "Dataset configuration must be a YAML mapping"
        raise InvalidDatasetConfigError(message) from exc
