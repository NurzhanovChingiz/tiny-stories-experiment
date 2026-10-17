"""YAML configuration for text normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from tiny_story.domain.errors import InvalidNormalizeTextConfigError
from tiny_story.infrastructure.config.yaml_config_repository import YamlConfigRepository

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class NormalizeTextConfig:
    """Validated settings for one text-normalization run.

    Attributes:
        enabled: Whether normalization runs.
        save_processed: Whether normalized text is written.
        sentence_delimiter: One-character marker inserted after . ! ?.
    """

    enabled: bool
    save_processed: bool
    sentence_delimiter: str


def parse_normalize_text_config(document: dict[str, object]) -> NormalizeTextConfig:
    """Validate a normalization mapping.

    Args:
        document: YAML mapping for text normalization.

    Returns:
        The validated configuration.

    Raises:
        InvalidNormalizeTextConfigError: A field is missing, mistyped, or inconsistent.

    Flow:
        1. Require the boolean flags.
        2. Require a one-character sentence delimiter.
        3. Reject saving processed text when normalization is off.
        4. Return the configuration.
    """
    # 1. Require the boolean flags.
    enabled = _flag(document.get("enabled"), "enabled")
    save_processed = _flag(document.get("save_processed"), "save_processed")

    # 2. Require a one-character sentence delimiter.
    sentence_delimiter = document.get("sentence_delimiter")
    if not isinstance(sentence_delimiter, str) or len(sentence_delimiter) != 1:
        message = "sentence_delimiter must be one character"
        raise InvalidNormalizeTextConfigError(message)

    # 3. Reject saving processed text when normalization is off.
    if save_processed and not enabled:
        message = "save_processed requires enabled to be true"
        raise InvalidNormalizeTextConfigError(message)

    # 4. Return the configuration.
    return NormalizeTextConfig(
        enabled=enabled,
        save_processed=save_processed,
        sentence_delimiter=sentence_delimiter,
    )


def load_normalize_text_config(path: Path) -> NormalizeTextConfig:
    """Load and validate a text-normalization configuration file.

    Args:
        path: YAML text-normalization configuration.

    Returns:
        The parsed configuration.

    Raises:
        InvalidNormalizeTextConfigError: The file is missing, unreadable, or invalid.

    Flow:
        1. Reject a missing or unreadable configuration file.
        2. Load the mapping and report read, decode, YAML, and shape failures.
    """
    # 1. Reject a missing or unreadable configuration file.
    try:
        exists = path.is_file()
    except OSError as exc:
        message = f"Could not read normalize text configuration: {path}"
        raise InvalidNormalizeTextConfigError(message) from exc
    if not exists:
        message = f"Normalize text configuration file does not exist: {path}"
        raise InvalidNormalizeTextConfigError(message)

    # 2. Load the mapping and report read, decode, YAML, and shape failures.
    repository = YamlConfigRepository(
        {NormalizeTextConfig: path},
        {NormalizeTextConfig: parse_normalize_text_config},
    )
    try:
        return repository.get(NormalizeTextConfig)
    except (OSError, UnicodeError) as exc:
        message = f"Could not read normalize text configuration: {path}"
        raise InvalidNormalizeTextConfigError(message) from exc
    except yaml.YAMLError as exc:
        message = f"Normalize text configuration is not valid YAML: {path}"
        raise InvalidNormalizeTextConfigError(message) from exc
    except TypeError as exc:
        message = "Normalize text configuration must be a YAML mapping"
        raise InvalidNormalizeTextConfigError(message) from exc


def _flag(value: object, field: str) -> bool:
    """Return a boolean configuration flag.

    Args:
        value: Candidate flag value.
        field: Field name used in the error.

    Returns:
        The boolean value.

    Raises:
        InvalidNormalizeTextConfigError: The value is not a boolean.
    """
    if not isinstance(value, bool):
        message = f"{field} must be true or false"
        raise InvalidNormalizeTextConfigError(message)
    return value
