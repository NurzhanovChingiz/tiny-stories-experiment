"""Tests for text-normalization configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tiny_story.domain.errors import InvalidNormalizeTextConfigError
from tiny_story.infrastructure.config.normalize_text_config import (
    load_normalize_text_config,
)

_VALID = """\
enabled: true
save_processed: false
sentence_delimiter: '|'
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


def test_load_normalize_text_config_reads_valid_file(tmp_path: Path) -> None:
    """A valid file keeps the flags and the sentence delimiter."""
    path = _write(tmp_path / "normalize_text.yaml", _VALID)

    config = load_normalize_text_config(path)

    assert config.enabled is True
    assert config.save_processed is False
    assert config.sentence_delimiter == "|"


def test_load_normalize_text_config_reads_repository_file() -> None:
    """The repository normalization configuration loads."""
    config = load_normalize_text_config(Path("config/normalize_text.yaml"))

    assert config.enabled is True
    assert config.save_processed is True
    assert config.sentence_delimiter == "|"


def test_load_normalize_text_config_rejects_empty_sentence_delimiter(
    tmp_path: Path,
) -> None:
    """An empty sentence delimiter is rejected."""
    content = _VALID.replace("sentence_delimiter: '|'", 'sentence_delimiter: ""')
    path = _write(tmp_path / "normalize_text.yaml", content)

    with pytest.raises(InvalidNormalizeTextConfigError, match="sentence_delimiter"):
        load_normalize_text_config(path)


def test_load_normalize_text_config_rejects_multi_char_sentence_delimiter(
    tmp_path: Path,
) -> None:
    """A sentence delimiter longer than one character is rejected."""
    content = _VALID.replace("sentence_delimiter: '|'", "sentence_delimiter: '||'")
    path = _write(tmp_path / "normalize_text.yaml", content)

    with pytest.raises(InvalidNormalizeTextConfigError, match="sentence_delimiter"):
        load_normalize_text_config(path)


def test_load_normalize_text_config_rejects_save_processed_when_disabled(
    tmp_path: Path,
) -> None:
    """Saving processed text requires normalization to be enabled."""
    content = _VALID.replace("enabled: true", "enabled: false").replace(
        "save_processed: false",
        "save_processed: true",
    )
    path = _write(tmp_path / "normalize_text.yaml", content)

    with pytest.raises(
        InvalidNormalizeTextConfigError, match="save_processed requires enabled"
    ):
        load_normalize_text_config(path)


def test_load_normalize_text_config_rejects_non_utf8_bytes(tmp_path: Path) -> None:
    """Bytes that are not UTF-8 are reported as an invalid configuration."""
    path = tmp_path / "normalize_text.yaml"
    path.write_bytes(b"enabled: true\xff")

    with pytest.raises(InvalidNormalizeTextConfigError, match="Could not read"):
        load_normalize_text_config(path)


def test_load_normalize_text_config_rejects_an_unreadable_directory(
    tmp_path: Path,
) -> None:
    """A configuration that cannot be stat'd is reported as unreadable."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory mode 000")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    path = blocked / "normalize_text.yaml"
    path.write_text(_VALID, encoding="utf-8")
    blocked.chmod(0)
    try:
        with pytest.raises(InvalidNormalizeTextConfigError, match="Could not read"):
            load_normalize_text_config(path)
    finally:
        blocked.chmod(0o700)
