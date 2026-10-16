"""Behavior of the .jsonl text reader."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from tiny_story.application.ports.text_source import TextSource
from tiny_story.domain.entities.text import Text
from tiny_story.infrastructure.io.read_text_jsonl import ReadTextJsonl

if TYPE_CHECKING:
    from pathlib import Path


def _jsonl(*bodies: str) -> str:
    """Return JSONL lines whose ``text`` fields are ``bodies``.

    Args:
        bodies: Story strings, in file order.

    Returns:
        One JSON object per line, ending with a newline.
    """
    return "".join(json.dumps({"text": body}) + "\n" for body in bodies)


@pytest.mark.parametrize("suffix", [".jsonl", ".JSONL"])
def test_read_returns_each_text_field(tmp_path: Path, suffix: str) -> None:
    """A .jsonl file returns one text per object, in file order."""
    path = tmp_path / f"stories{suffix}"
    path.write_text(_jsonl("Once upon a time.", "The end."), encoding="utf-8")

    assert ReadTextJsonl().read(path) == (
        Text(body="Once upon a time."),
        Text(body="The end."),
    )


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    """A blank line between objects is not a record."""
    path = tmp_path / "stories.jsonl"
    path.write_text(_jsonl("one") + "\n" + _jsonl("two"), encoding="utf-8")

    assert ReadTextJsonl().read(path) == (Text(body="one"), Text(body="two"))


def test_read_returns_no_text_for_an_empty_file(tmp_path: Path) -> None:
    """An empty .jsonl file has no text values."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert ReadTextJsonl().read(path) == ()


def test_read_keeps_an_empty_text_field(tmp_path: Path) -> None:
    """An empty string is a text value."""
    path = tmp_path / "stories.jsonl"
    path.write_text(_jsonl(""), encoding="utf-8")

    assert ReadTextJsonl().read(path) == (Text(body=""),)


@pytest.mark.parametrize("suffix", [".txt", ".md", ".csv", ""])
def test_read_rejects_a_non_jsonl_extension(tmp_path: Path, suffix: str) -> None:
    """Only a .jsonl suffix is accepted."""
    path = tmp_path / f"stories{suffix}"
    path.write_text(_jsonl("hello"), encoding="utf-8")

    with pytest.raises(
        ValueError, match=f"Unsupported file extension: {re.escape(suffix)}"
    ):
        ReadTextJsonl().read(path)


def test_read_rejects_a_line_that_is_not_an_object(tmp_path: Path) -> None:
    """A JSON array is not a story record."""
    path = tmp_path / "stories.jsonl"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TypeError, match="JSONL line must be an object"):
        ReadTextJsonl().read(path)


@pytest.mark.parametrize("line", ['{"summary": "x"}\n', '{"text": 1}\n'])
def test_read_rejects_a_record_without_string_text(tmp_path: Path, line: str) -> None:
    """Each object must contain a string text field."""
    path = tmp_path / "stories.jsonl"
    path.write_text(line, encoding="utf-8")

    with pytest.raises(
        TypeError, match="JSONL object must contain a string text field"
    ):
        ReadTextJsonl().read(path)


def test_read_rejects_a_line_that_is_not_json(tmp_path: Path) -> None:
    """A line that is not JSON is not a story record."""
    path = tmp_path / "stories.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        ReadTextJsonl().read(path)


def test_readers_read_only_the_path_they_are_given(tmp_path: Path) -> None:
    """Each call returns the file passed to that call."""
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_path.write_text(_jsonl("first"), encoding="utf-8")
    second_path.write_text(_jsonl("second"), encoding="utf-8")
    first = ReadTextJsonl()
    second = ReadTextJsonl()

    assert first.read(first_path) == (Text(body="first"),)
    assert second.read(second_path) == (Text(body="second"),)
    assert isinstance(first, TextSource)


def test_read_missing_file_raises(tmp_path: Path) -> None:
    """A missing .jsonl path is not turned into text."""
    with pytest.raises(FileNotFoundError):
        ReadTextJsonl().read(tmp_path / "missing.jsonl")


def test_read_rejects_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    """A .jsonl file that is not UTF-8 is not turned into text."""
    path = tmp_path / "stories.jsonl"
    path.write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        ReadTextJsonl().read(path)
