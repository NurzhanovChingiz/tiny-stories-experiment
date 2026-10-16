"""YAML configuration for text cleaning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import regex as re
import yaml

from tiny_story.domain.errors import InvalidCleanTextConfigError
from tiny_story.infrastructure.config.yaml_config_repository import YamlConfigRepository
from tiny_story.infrastructure.text_processing.clean_text.compiled_patterns import (
    REGEX_TIMEOUT_SECONDS,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class CleanTextConfig:
    """Validated settings for one text-cleaning run.

    Attributes:
        enabled: Whether cleaning runs.
        save_processed: Whether cleaned text is written.
        page_header_pattern: Regular expression for page-header lines.
        junk_line_pattern: Regular expression for junk lines.
        reflow_hard_wrapped_lines: Whether to join lines broken mid-sentence.
        join_hyphenated_line_breaks: Whether to join hyphenated line breaks.
    """

    enabled: bool
    save_processed: bool
    page_header_pattern: str
    junk_line_pattern: str
    reflow_hard_wrapped_lines: bool
    join_hyphenated_line_breaks: bool


def parse_clean_text_config(document: dict[str, object]) -> CleanTextConfig:
    """Validate a cleaning mapping.

    Args:
        document: YAML mapping for text cleaning.

    Returns:
        The validated configuration.

    Raises:
        InvalidCleanTextConfigError: A field is missing, mistyped, or inconsistent.

    Flow:
        1. Require the boolean flags and pattern strings.
        2. Reject saving processed text when cleaning is off.
        3. Reject a pattern that does not compile or times out.
        4. Return the configuration.
    """
    # 1. Require the boolean flags and pattern strings.
    enabled = _flag(document.get("enabled"), "enabled")
    save_processed = _flag(document.get("save_processed"), "save_processed")
    reflow = _flag(
        document.get("reflow_hard_wrapped_lines"), "reflow_hard_wrapped_lines"
    )
    join_breaks = _flag(
        document.get("join_hyphenated_line_breaks"), "join_hyphenated_line_breaks"
    )
    page_header_pattern = _pattern(
        document.get("page_header_pattern"), "page_header_pattern"
    )
    junk_line_pattern = _pattern(document.get("junk_line_pattern"), "junk_line_pattern")

    # 2. Reject saving processed text when cleaning is off.
    if save_processed and not enabled:
        message = "save_processed requires enabled to be true"
        raise InvalidCleanTextConfigError(message)

    # 3. Reject a pattern that does not compile or times out.
    _require_regex(page_header_pattern, "page_header_pattern")
    _require_regex(junk_line_pattern, "junk_line_pattern")

    # 4. Return the configuration.
    return CleanTextConfig(
        enabled=enabled,
        save_processed=save_processed,
        page_header_pattern=page_header_pattern,
        junk_line_pattern=junk_line_pattern,
        reflow_hard_wrapped_lines=reflow,
        join_hyphenated_line_breaks=join_breaks,
    )


def load_clean_text_config(path: Path) -> CleanTextConfig:
    """Load and validate a text-cleaning configuration file.

    Args:
        path: YAML text-cleaning configuration.

    Returns:
        The parsed configuration.

    Raises:
        InvalidCleanTextConfigError: The file is missing, unreadable, or invalid.

    Flow:
        1. Reject a missing configuration file.
        2. Load the mapping and report read, YAML, and shape failures.
    """
    # 1. Reject a missing configuration file.
    if not path.is_file():
        message = f"Clean text configuration file does not exist: {path}"
        raise InvalidCleanTextConfigError(message)

    # 2. Load the mapping and report read, YAML, and shape failures.
    repository = YamlConfigRepository(
        {CleanTextConfig: path},
        {CleanTextConfig: parse_clean_text_config},
    )
    try:
        return repository.get(CleanTextConfig)
    except OSError as exc:
        message = f"Could not read clean text configuration: {path}"
        raise InvalidCleanTextConfigError(message) from exc
    except yaml.YAMLError as exc:
        message = f"Clean text configuration is not valid YAML: {path}"
        raise InvalidCleanTextConfigError(message) from exc
    except TypeError as exc:
        message = "Clean text configuration must be a YAML mapping"
        raise InvalidCleanTextConfigError(message) from exc


def _flag(value: object, field: str) -> bool:
    """Return a boolean configuration flag.

    Args:
        value: Candidate flag value.
        field: Field name used in the error.

    Returns:
        The boolean value.

    Raises:
        InvalidCleanTextConfigError: The value is not a boolean.
    """
    if not isinstance(value, bool):
        message = f"{field} must be true or false"
        raise InvalidCleanTextConfigError(message)
    return value


def _pattern(value: object, field: str) -> str:
    """Return a pattern string.

    Args:
        value: Candidate pattern value.
        field: Field name used in the error.

    Returns:
        The pattern string.

    Raises:
        InvalidCleanTextConfigError: The value is not a string.
    """
    if not isinstance(value, str):
        message = f"{field} must be a string"
        raise InvalidCleanTextConfigError(message)
    return value


def _require_regex(pattern: str, field: str) -> None:
    """Reject a pattern that does not compile or times out on an empty string.

    Args:
        pattern: Regular expression text.
        field: Field name used in the error.

    Returns:
        None.

    Raises:
        InvalidCleanTextConfigError: The pattern is invalid or timed out.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        message = f"{field} is not a valid regular expression: {exc}"
        raise InvalidCleanTextConfigError(message) from exc
    try:
        compiled.search("", timeout=REGEX_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        message = f"{field} regular expression timed out during validation"
        raise InvalidCleanTextConfigError(message) from exc
