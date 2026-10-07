"""CLI progress lines and artifact writing for tokenizer training."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest
from loguru import logger
from tokenizers import Tokenizer
from typer.testing import CliRunner

from tiny_stories_experiment.application.ports.tokenizer_repository import (
    TOKENIZER_FILENAME,
)
from tiny_stories_experiment.composition import (
    DEFAULT_TOKENIZER_DESTINATION,
    DEFAULT_TOKENIZER_SPEC,
)
from tiny_stories_experiment.entrypoints.cli.__main__ import app
from tiny_stories_experiment.entrypoints.cli.commands.train_tokenizer import (
    train_tokenizer_command,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

STORIES = (
    "alpha alpha alpha.",
    "alpha beta alpha.",
    "beta beta beta.",
)


@pytest.fixture
def restored_loguru() -> Iterator[None]:
    """Put the default stderr sink back after the command replaces it."""
    yield
    logger.remove()
    logger.add(sys.stderr)


def _write_fixture(processed: Path) -> dict[str, bytes]:
    """Write both prepared splits and return their bytes.

    Args:
        processed: Directory for ``train.jsonl`` and ``valid.jsonl``.

    Returns:
        File contents keyed by name, recorded before training.
    """
    processed.mkdir(parents=True)
    contents: dict[str, bytes] = {}
    for name in ("train.jsonl", "valid.jsonl"):
        lines = [json.dumps({"text": story}) for story in STORIES]
        payload = ("\n".join(lines) + "\n").encode()
        (processed / name).write_bytes(payload)
        contents[name] = payload
    return contents


def test_default_tokenizer_directory_is_artifacts_tokenizers() -> None:
    """Trained tokenizers default to artifacts/tokenizers."""
    assert DEFAULT_TOKENIZER_DESTINATION.parts[-2:] == ("artifacts", "tokenizers")
    assert DEFAULT_TOKENIZER_SPEC.vocab_size == 4096
    assert DEFAULT_TOKENIZER_SPEC.special_tokens == ("<|endoftext|>",)


def test_train_tokenizer_command_writes_artifact_without_changing_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    restored_loguru: None,
) -> None:
    """The command stores tokenizer.json and leaves the prepared JSONL bytes."""
    del restored_loguru
    processed = tmp_path / "processed"
    artifacts = tmp_path / "tokenizers"
    before = _write_fixture(processed)

    train_tokenizer_command(
        processed,
        artifacts,
        name="tiny_stories_bpe",
        vocab_size=300,
        min_frequency=2,
    )
    captured = capsys.readouterr()
    artifact = artifacts / "tiny_stories_bpe" / TOKENIZER_FILENAME
    tokenizer = Tokenizer.from_file(str(artifact))

    assert captured.out == ""
    assert {name: (processed / name).read_bytes() for name in before} == before
    assert len(tokenizer.encode("alpha").ids) == 1
    assert captured.err.splitlines() == [
        "tokenizer tiny_stories_bpe",
        f"vocab {tokenizer.get_vocab_size()}",
        f"stories {len(STORIES) * 2}",
        f"artifact {artifact}",
        f"processed directory {processed}",
    ]


def test_cli_train_tokenizer_writes_the_artifact(
    tmp_path: Path,
    restored_loguru: None,
) -> None:
    """The registered train-tokenizer command stores tokenizer.json."""
    del restored_loguru
    processed = tmp_path / "processed"
    artifacts = tmp_path / "tokenizers"
    before = _write_fixture(processed)

    result = CliRunner().invoke(
        app,
        [
            "train-tokenizer",
            "--processed-directory",
            str(processed),
            "--artifact-directory",
            str(artifacts),
            "--vocab-size",
            "300",
            "--min-frequency",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert (artifacts / "tiny_stories_bpe" / TOKENIZER_FILENAME).is_file()
    assert {name: (processed / name).read_bytes() for name in before} == before
