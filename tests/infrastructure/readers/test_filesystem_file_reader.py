"""Behavior of the filesystem file reader."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from tiny_story.infrastructure.readers.filesystem_file_reader import (
    FilesystemFileReader,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("suffix", [".txt", ".md", ".csv", ".TXT"])
def test_read_returns_utf8_text(tmp_path: Path, suffix: str) -> None:
    """Text suffixes return the file's UTF-8 contents."""
    path = tmp_path / f"note{suffix}"
    path.write_text("hello\n", encoding="utf-8")

    assert FilesystemFileReader().read(path) == "hello\n"


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".wav", ".flac", ".bin"])
def test_read_rejects_an_unsupported_extension(tmp_path: Path, suffix: str) -> None:
    """Image, audio, and other non-text suffixes are rejected."""
    path = tmp_path / f"payload{suffix}"
    path.write_bytes(b"data")

    with pytest.raises(
        ValueError, match=f"Unsupported file extension: {re.escape(suffix)}"
    ):
        FilesystemFileReader().read(path)


def test_readers_read_only_the_path_they_are_given(tmp_path: Path) -> None:
    """Each call returns the file passed to that call."""
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("first", encoding="utf-8")
    second_path.write_text("second", encoding="utf-8")
    first = FilesystemFileReader()
    second = FilesystemFileReader()

    assert first.read(first_path) == "first"
    assert second.read(second_path) == "second"
