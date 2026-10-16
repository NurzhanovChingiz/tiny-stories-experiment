"""Tests for text-cleaning configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_story.domain.errors import InvalidCleanTextConfigError
from tiny_story.infrastructure.config.clean_text_config import load_clean_text_config

_VALID = """\
enabled: true
save_processed: false
page_header_pattern: '^Page'
junk_line_pattern: '^[/\\\\-]+$'
reflow_hard_wrapped_lines: true
join_hyphenated_line_breaks: false
"""


def _write(path: Path, content: str) -> Path:
    """Write a configuration fixture and return its path.

    Args:
        path: Destination file.
        content: YAML text.

    Returns:
        The path that was written.
    """
    path.write_text(content, encoding="utf-8")
    return path


def test_load_clean_text_config_reads_repository_file() -> None:
    """The repository cleaning configuration loads."""
    config = load_clean_text_config(Path("config/clean_text.yaml"))

    assert config.enabled is True
    assert config.save_processed is True
    assert config.join_hyphenated_line_breaks is True


def test_load_clean_text_config_rejects_invalid_regex(tmp_path: Path) -> None:
    """An unclosed pattern is rejected."""
    content = _VALID.replace(
        "page_header_pattern: '^Page'",
        "page_header_pattern: '[unclosed'",
    )
    path = _write(tmp_path / "clean_text.yaml", content)

    with pytest.raises(
        InvalidCleanTextConfigError, match="not a valid regular expression"
    ):
        load_clean_text_config(path)
