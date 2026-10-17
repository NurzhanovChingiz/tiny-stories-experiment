"""JSONL story tokenization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_story.domain.errors import TokenizeTextError
from tiny_story.infrastructure.text_processing.tokenize_text.jsonl_story_tokenizer import (
    JsonlStoryTokenizer,
)


def test_tokenize_files_adds_ids_and_one_shared_vocabulary(tmp_path: Path) -> None:
    """Two story files share a sorted vocabulary and keep their other fields."""
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text(
        '{"text": "hello world .|", "story_id": 7}\n',
        encoding="utf-8",
    )
    valid.write_text('{"text": "hello .|"}\n\n', encoding="utf-8")
    output = tmp_path / "tokenized"
    tokenizer = JsonlStoryTokenizer(enabled=True)

    tokenizer.tokenize_files(
        (train, valid),
        (output / "train.jsonl", output / "valid.jsonl"),
        output / "vocab.json",
    )

    train_story = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    valid_story = json.loads((output / "valid.jsonl").read_text(encoding="utf-8"))
    vocabulary = json.loads((output / "vocab.json").read_text(encoding="utf-8"))
    assert train_story["story_id"] == 7
    assert train_story["text"] == "hello world .|"
    assert train_story["token_ids"] == [
        vocabulary["<bos>"],
        vocabulary["hello"],
        vocabulary["world"],
        vocabulary[".|"],
        vocabulary["<eos>"],
    ]
    assert valid_story["token_ids"][1] == vocabulary["hello"]
    assert train.read_text(encoding="utf-8").startswith('{"text": "hello world .|"')


def test_tokenize_files_refuses_a_missing_source(tmp_path: Path) -> None:
    """A missing story file is not invented."""
    tokenizer = JsonlStoryTokenizer(enabled=True)

    with pytest.raises(TokenizeTextError, match="Story file does not exist"):
        tokenizer.tokenize_files(
            (tmp_path / "missing.jsonl",),
            (tmp_path / "out.jsonl",),
            tmp_path / "vocab.json",
        )

    assert not (tmp_path / "out.jsonl").exists()
    assert not (tmp_path / "vocab.json").exists()


def test_tokenize_files_leaves_an_existing_file_when_json_is_invalid(
    tmp_path: Path,
) -> None:
    """A bad line does not replace a destination that is already there."""
    source = tmp_path / "valid.jsonl"
    source.write_text('{"text": "ok"}\n[]\n', encoding="utf-8")
    destination = tmp_path / "out" / "valid.jsonl"
    destination.parent.mkdir()
    destination.write_text("KEEP\n", encoding="utf-8")
    tokenizer = JsonlStoryTokenizer(enabled=True)

    with pytest.raises(TokenizeTextError, match="JSONL line 2 must be an object"):
        tokenizer.tokenize_files(
            (source,),
            (destination,),
            tmp_path / "out" / "vocab.json",
        )

    assert destination.read_text(encoding="utf-8") == "KEEP\n"


def test_tokenize_files_parses_each_story_line_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A story line is decoded once, then reused for the vocabulary and the ids."""
    source = tmp_path / "stories.jsonl"
    source.write_text(
        '{"text": "one two"}\n\n{"text": "two three"}\n',
        encoding="utf-8",
    )
    parse_count = 0
    read_count = 0
    original_loads = json.loads
    original_read_text = Path.read_text

    def counting_loads(line: str) -> object:
        nonlocal parse_count
        parse_count += 1
        return original_loads(line)

    def counting_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        nonlocal read_count
        if self == source:
            read_count += 1
        return original_read_text(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(
        "tiny_story.infrastructure.text_processing.tokenize_text.jsonl_story_tokenizer.json.loads",
        counting_loads,
    )
    monkeypatch.setattr(Path, "read_text", counting_read_text)
    tokenizer = JsonlStoryTokenizer(enabled=True)

    tokenizer.tokenize_files(
        (source,),
        (tmp_path / "out.jsonl",),
        tmp_path / "vocab.json",
    )

    story_lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert parse_count == 2
    assert read_count == 1
    assert len(story_lines) == 2
    assert "token_ids" in json.loads(story_lines[0])
