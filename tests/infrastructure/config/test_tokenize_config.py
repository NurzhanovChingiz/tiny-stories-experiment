"""Tests for tokenization configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_story.domain.errors import InvalidTokenizeConfigError
from tiny_story.infrastructure.config.tokenize_config import load_tokenize_config

_VALID = """\
enabled: true
save_processed: true
"""


def test_load_tokenize_config_reads_repository_file() -> None:
    """The repository tokenization configuration loads."""
    config = load_tokenize_config(Path("config/tokenize.yaml"))

    assert config.enabled is True
    assert config.save_processed is True


def test_load_tokenize_config_rejects_save_processed_when_disabled(
    tmp_path: Path,
) -> None:
    """Saving tokenized text while tokenization is off is rejected."""
    path = tmp_path / "tokenize.yaml"
    path.write_text("enabled: false\nsave_processed: true\n", encoding="utf-8")

    with pytest.raises(InvalidTokenizeConfigError, match="save_processed"):
        load_tokenize_config(path)


def test_load_tokenize_config_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing configuration file is rejected."""
    with pytest.raises(InvalidTokenizeConfigError, match="does not exist"):
        load_tokenize_config(tmp_path / "missing.yaml")
