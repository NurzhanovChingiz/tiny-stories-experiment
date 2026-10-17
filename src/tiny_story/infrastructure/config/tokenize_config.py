"""YAML configuration for whitespace tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from tiny_story.domain.errors import InvalidTokenizeConfigError
from tiny_story.infrastructure.config.yaml_config_repository import YamlConfigRepository

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class TokenizeConfig:
    """Validated settings for one tokenization run.

    Attributes:
        enabled: Whether stories are encoded as token ids.
        save_processed: Whether tokenized files and the vocabulary are written.
    """

    enabled: bool
    save_processed: bool


def parse_tokenize_config(document: dict[str, object]) -> TokenizeConfig:
    """Validate a tokenization mapping.

    Args:
        document: YAML mapping for tokenization.

    Returns:
        The validated configuration.

    Raises:
        InvalidTokenizeConfigError: A field is missing, mistyped, or inconsistent.

    Flow:
        1. Require the boolean flags.
        2. Reject saving tokenized text when tokenization is off.
        3. Return the configuration.
    """
    # 1. Require the boolean flags.
    enabled = _flag(document.get("enabled"), "enabled")
    save_processed = _flag(document.get("save_processed"), "save_processed")

    # 2. Reject saving tokenized text when tokenization is off.
    if save_processed and not enabled:
        message = "save_processed requires enabled to be true"
        raise InvalidTokenizeConfigError(message)

    # 3. Return the configuration.
    return TokenizeConfig(enabled=enabled, save_processed=save_processed)


def load_tokenize_config(path: Path) -> TokenizeConfig:
    """Load and validate a tokenization configuration file.

    Args:
        path: YAML tokenization configuration.

    Returns:
        The parsed configuration.

    Raises:
        InvalidTokenizeConfigError: The file is missing, unreadable, or invalid.

    Flow:
        1. Reject a missing or unreadable configuration file.
        2. Load the mapping and report read, decode, YAML, and shape failures.
    """
    # 1. Reject a missing or unreadable configuration file.
    try:
        exists = path.is_file()
    except OSError as exc:
        message = f"Could not read tokenize configuration: {path}"
        raise InvalidTokenizeConfigError(message) from exc
    if not exists:
        message = f"Tokenize configuration file does not exist: {path}"
        raise InvalidTokenizeConfigError(message)

    # 2. Load the mapping and report read, decode, YAML, and shape failures.
    repository = YamlConfigRepository(
        {TokenizeConfig: path},
        {TokenizeConfig: parse_tokenize_config},
    )
    try:
        return repository.get(TokenizeConfig)
    except (OSError, UnicodeError) as exc:
        message = f"Could not read tokenize configuration: {path}"
        raise InvalidTokenizeConfigError(message) from exc
    except yaml.YAMLError as exc:
        message = f"Tokenize configuration is not valid YAML: {path}"
        raise InvalidTokenizeConfigError(message) from exc
    except TypeError as exc:
        message = "Tokenize configuration must be a YAML mapping"
        raise InvalidTokenizeConfigError(message) from exc


def _flag(value: object, field: str) -> bool:
    """Return a boolean configuration flag.

    Args:
        value: Candidate flag value.
        field: Field name used in the error.

    Returns:
        The boolean value.

    Raises:
        InvalidTokenizeConfigError: The value is not a boolean.
    """
    if not isinstance(value, bool):
        message = f"{field} must be true or false"
        raise InvalidTokenizeConfigError(message)
    return value
