"""YAML schema for Hugging Face dataset acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tiny_story.application.use_cases.download_dataset import ExistingFilePolicy
from tiny_story.domain.dataset.dataset_spec import DatasetSpec, require_dataset_spec
from tiny_story.domain.errors import (
    InvalidDatasetConfigError,
    InvalidDatasetSpecError,
    InvalidRawFilenameError,
)
from tiny_story.infrastructure.config.yaml_config_repository import YamlConfigRepository

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class HuggingFaceSourceConfig:
    """Hugging Face dataset source settings.

    Attributes:
        repository: Dataset id such as ``owner/name``.
        revision: Branch, tag, or commit to download.
        cache_dir: Interim Hugging Face cache directory.
    """

    repository: str
    revision: str
    cache_dir: Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Acquisition settings loaded from YAML.

    Attributes:
        spec: Dataset identity, including the raw file names to store.
        destination: Directory that receives raw files.
        existing: Whether a complete raw file is kept or acquired again.
        source: Hugging Face repository, revision, and cache directory.
    """

    spec: DatasetSpec
    destination: Path
    existing: ExistingFilePolicy
    source: HuggingFaceSourceConfig


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load one dataset configuration file.

    Args:
        path: YAML file to read.

    Returns:
        The parsed dataset configuration.

    Raises:
        InvalidDatasetConfigError: The file is missing, unreadable, or invalid.

    Flow:
        1. Reject a missing configuration file.
        2. Parse the YAML mapping into a dataset configuration.
    """
    # 1. Reject a missing configuration file.
    if not path.is_file():
        message = f"Dataset configuration file does not exist: {path}"
        raise InvalidDatasetConfigError(message)

    # 2. Parse the YAML mapping into a dataset configuration.
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


def parse_dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    """Build a dataset configuration from a YAML mapping.

    Args:
        raw: Mapping loaded from YAML.

    Returns:
        The dataset configuration.

    Raises:
        InvalidDatasetConfigError: A required field is missing or invalid.

    Flow:
        1. Read the dataset identity.
        2. Read the destination, existing-file policy, and Hugging Face source.
    """
    # 1. Read the dataset identity.
    dataset = _mapping(raw.get("dataset"), "dataset")
    spec = _spec(dataset)

    # 2. Read the destination, existing-file policy, and Hugging Face source.
    return DatasetConfig(
        spec=spec,
        destination=_destination(dataset),
        existing=_existing_policy(dataset),
        source=_huggingface_source(_mapping(dataset.get("source"), "dataset.source")),
    )


def _spec(dataset: Mapping[str, object]) -> DatasetSpec:
    """Return the dataset identity from the dataset mapping.

    Args:
        dataset: YAML ``dataset`` mapping.

    Returns:
        A dataset spec that can be stored as raw files.

    Raises:
        InvalidDatasetConfigError: The name or file list is invalid.
    """
    try:
        return require_dataset_spec(
            DatasetSpec(
                name=_text(dataset.get("name"), "dataset.name"),
                filenames=_filenames(dataset),
            ),
        )
    except (InvalidDatasetSpecError, InvalidRawFilenameError) as exc:
        message = str(exc)
        raise InvalidDatasetConfigError(message) from exc


def _filenames(dataset: Mapping[str, object]) -> tuple[str, ...]:
    """Return the raw file names listed for the dataset.

    Args:
        dataset: YAML ``dataset`` mapping.

    Returns:
        The file names in listed order.

    Raises:
        InvalidDatasetConfigError: ``dataset.files`` is not a list of strings.
    """
    raw_files = dataset.get("files")
    if not isinstance(raw_files, list) or len(raw_files) == 0:
        message = "dataset.files must list each raw file name once"
        raise InvalidDatasetConfigError(message)
    names: list[str] = []
    for item in raw_files:
        if not isinstance(item, str):
            message = "dataset.files must list each raw file name once"
            raise InvalidDatasetConfigError(message)
        names.append(item)
    return tuple(names)


def _destination(dataset: Mapping[str, object]) -> Path:
    """Return the raw destination directory.

    Args:
        dataset: YAML ``dataset`` mapping.

    Returns:
        The destination path.

    Raises:
        InvalidDatasetConfigError: The destination is missing or blank.
    """
    return Path(_text(dataset.get("destination"), "dataset.destination"))


def _existing_policy(dataset: Mapping[str, object]) -> ExistingFilePolicy:
    """Return whether complete raw files are kept.

    Args:
        dataset: YAML ``dataset`` mapping.

    Returns:
        ``KEEP`` unless ``overwrite`` is true.

    Raises:
        InvalidDatasetConfigError: ``overwrite`` is present and not a boolean.
    """
    if "overwrite" not in dataset:
        return ExistingFilePolicy.KEEP
    raw = dataset["overwrite"]
    if isinstance(raw, bool):
        if raw:
            return ExistingFilePolicy.REPLACE
        return ExistingFilePolicy.KEEP
    message = "dataset.overwrite must be true or false"
    raise InvalidDatasetConfigError(message)


def _huggingface_source(source: Mapping[str, object]) -> HuggingFaceSourceConfig:
    """Return Hugging Face source settings.

    Args:
        source: YAML source mapping.

    Returns:
        Repository, revision, and cache directory.

    Raises:
        InvalidDatasetConfigError: The repository id or another field is invalid.
    """
    repository = _text(source.get("repository"), "dataset.source.repository")
    owner, separator, name = repository.partition("/")
    if separator != "/" or owner == "" or name == "" or "/" in name:
        message = (
            "dataset.source.repository must be a Hugging Face dataset id "
            "such as owner/name"
        )
        raise InvalidDatasetConfigError(message)
    revision = _optional_text(source, "revision", "main", "dataset.source.revision")
    cache_dir = _optional_path(
        source,
        "cache_dir",
        "data/interim/huggingface",
        "dataset.source.cache_dir",
    )
    return HuggingFaceSourceConfig(
        repository=repository,
        revision=revision,
        cache_dir=cache_dir,
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    """Return a string-keyed mapping.

    Args:
        value: Candidate mapping.
        label: Field path used in the error message.

    Returns:
        A copy with string keys.

    Raises:
        InvalidDatasetConfigError: The value is not a mapping with string keys.
    """
    if not isinstance(value, dict):
        message = f"{label} must be a mapping"
        raise InvalidDatasetConfigError(message)
    mapped: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            message = f"{label} must use string keys"
            raise InvalidDatasetConfigError(message)
        mapped[key] = item
    return mapped


def _text(value: object, label: str) -> str:
    """Return a non-empty string without surrounding whitespace.

    Args:
        value: Candidate string.
        label: Field path used in the error message.

    Returns:
        The same string when it can be stored as configuration.

    Raises:
        InvalidDatasetConfigError: The value is missing, blank, or not a string.
    """
    if not isinstance(value, str) or value == "" or value.strip() != value:
        message = f"{label} must be a non-empty string"
        raise InvalidDatasetConfigError(message)
    return value


def _optional_text(
    source: Mapping[str, object],
    key: str,
    default: str,
    label: str,
) -> str:
    """Return a configured string or the default when the key is absent.

    Args:
        source: Mapping that may contain the key.
        key: Optional field name.
        default: Value used when the key is absent.
        label: Field path used when the value is invalid.

    Returns:
        The configured string or ``default``.
    """
    if key not in source:
        return default
    return _text(source[key], label)


def _optional_path(
    source: Mapping[str, object],
    key: str,
    default: str,
    label: str,
) -> Path:
    """Return a configured path or the default when the key is absent.

    Args:
        source: Mapping that may contain the key.
        key: Optional field name.
        default: Path used when the key is absent.
        label: Field path used when the value is invalid.

    Returns:
        The configured path or ``default``.
    """
    return Path(_optional_text(source, key, default, label))
