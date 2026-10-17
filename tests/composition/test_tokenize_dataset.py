"""Behavior of configured story tokenization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tiny_story.composition import tokenize_dataset
from tiny_story.domain.errors import TokenizeTextError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_default_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the default tokenize log in the test directory.

    Args:
        tmp_path: Per-test directory.
        monkeypatch: Pytest patch helper used to change the working directory.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)


def test_tokenize_dataset_writes_ids_outside_the_processed_file(tmp_path: Path) -> None:
    """Tokenization writes ids and a vocabulary, and leaves the processed file."""
    processed = tmp_path / "processed"
    processed.mkdir()
    source = processed / "valid.jsonl"
    original = '{"text": "hello world .|"}\n'
    source.write_text(original, encoding="utf-8")
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        f"""
dataset:
  name: other_corpus
  files:
    - valid.jsonl
  destination: {tmp_path / "raw"}
  overwrite: false
  source:
    repository: some-org/other-corpus
    revision: abc123
    cache_dir: {tmp_path / "cache"}
""",
        encoding="utf-8",
    )
    tokenize_config = tmp_path / "tokenize.yaml"
    tokenize_config.write_text(
        "enabled: true\nsave_processed: true\n",
        encoding="utf-8",
    )
    tokenized = tmp_path / "tokenized"

    written = tokenize_dataset(
        dataset_config,
        tokenize_config,
        input_dir=processed,
        output_dir=tokenized,
    )

    story = json.loads((tokenized / "valid.jsonl").read_text(encoding="utf-8"))
    vocabulary = json.loads((tokenized / "vocab.json").read_text(encoding="utf-8"))
    assert written == (tokenized / "valid.jsonl",)
    assert source.read_text(encoding="utf-8") == original
    assert story["token_ids"] == [
        vocabulary["<bos>"],
        vocabulary["hello"],
        vocabulary["world"],
        vocabulary[".|"],
        vocabulary["<eos>"],
    ]
    log_path = tmp_path / "artifacts" / "tokenize" / "logs" / "other_corpus.log"
    assert "Wrote tokenized file" in log_path.read_text(encoding="utf-8")


def test_tokenize_dataset_records_a_missing_processed_file(tmp_path: Path) -> None:
    """A missing processed file is recorded and is not invented."""
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        f"""
dataset:
  name: other_corpus
  files:
    - valid.jsonl
  destination: {tmp_path / "raw"}
  overwrite: false
  source:
    repository: some-org/other-corpus
    revision: abc123
    cache_dir: {tmp_path / "cache"}
""",
        encoding="utf-8",
    )
    tokenize_config = tmp_path / "tokenize.yaml"
    tokenize_config.write_text(
        "enabled: true\nsave_processed: true\n",
        encoding="utf-8",
    )

    with pytest.raises(TokenizeTextError, match="Story file does not exist"):
        tokenize_dataset(
            dataset_config,
            tokenize_config,
            input_dir=tmp_path / "processed",
            output_dir=tmp_path / "tokenized",
        )

    log_path = tmp_path / "artifacts" / "tokenize" / "logs" / "other_corpus.log"
    assert "Tokenization failed" in log_path.read_text(encoding="utf-8")
