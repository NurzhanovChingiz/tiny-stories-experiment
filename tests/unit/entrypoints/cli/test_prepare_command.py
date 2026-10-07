"""CLI progress lines for TinyStoriesV2 preparation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from loguru import logger
from typer.testing import CliRunner

from tiny_stories_experiment.application.results.prepare_outcome import (
    PreparedSplit,
    PrepareOutcome,
)
from tiny_stories_experiment.composition import (
    DEFAULT_PROCESSED_DESTINATION,
    TINY_STORIES_V2,
)
from tiny_stories_experiment.domain.dataset.text_sample import DatasetSplit
from tiny_stories_experiment.entrypoints.cli.__main__ import app
from tiny_stories_experiment.entrypoints.cli.commands.prepare import prepare_command

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def restored_loguru() -> Iterator[None]:
    """Put the default stderr sink back after the command replaces it."""
    yield
    logger.remove()
    logger.add(sys.stderr)


def test_default_processed_directory_is_data_processed_tiny_stories() -> None:
    """Derived files default to data/processed/tiny_stories."""
    assert DEFAULT_PROCESSED_DESTINATION.parts[-3:] == (
        "data",
        "processed",
        "tiny_stories",
    )


def test_prepare_command_logs_split_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    restored_loguru: None,
) -> None:
    """Progress lines name the dataset, each derived file, and both directories."""
    del restored_loguru
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    outcome = PrepareOutcome(
        raw_directory=raw,
        processed_directory=processed,
        splits=(
            PreparedSplit(
                raw_filename="TinyStoriesV2-GPT4-train.jsonl",
                split=DatasetSplit.TRAIN,
                story_count=2,
                output_name="train.jsonl",
            ),
            PreparedSplit(
                raw_filename="TinyStoriesV2-GPT4-valid.jsonl",
                split=DatasetSplit.VALID,
                story_count=1,
                output_name="valid.jsonl",
            ),
        ),
    )

    def fake_prepare(raw_directory: Path, processed_directory: Path) -> PrepareOutcome:
        assert raw_directory == raw
        assert processed_directory == processed
        return outcome

    monkeypatch.setattr(
        "tiny_stories_experiment.entrypoints.cli.commands.prepare.prepare_default_dataset",
        fake_prepare,
    )

    prepare_command(raw, processed)
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err.splitlines() == [
        f"dataset {TINY_STORIES_V2.source_url}",
        "prepared train.jsonl 2",
        "prepared valid.jsonl 1",
        f"raw directory {raw}",
        f"processed directory {processed}",
    ]


def test_cli_help_lists_download_prepare_and_train_tokenizer() -> None:
    """The package entry point exposes download, prepare, and train-tokenizer."""
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "download" in result.output
    assert "prepare" in result.output
    assert "train-tokenizer" in result.output
