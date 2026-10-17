"""Behavior of the tokenize command and its Makefile target."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tiny_story.entrypoints.cli.app import app
from tiny_story.entrypoints.cli.download import download
from tiny_story.entrypoints.cli.tokenize import tokenize

runner = CliRunner()


def test_tokenize_command_uses_the_shipped_configuration() -> None:
    """The tokenize command reads the shipped configuration unless flags override it."""
    seen: dict[str, Path | None] = {}

    def fake_tokenize(
        dataset_config_path: Path,
        tokenize_config_path: Path,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        log_path: Path | None = None,
    ) -> tuple[Path, ...]:
        seen["dataset"] = dataset_config_path
        seen["tokenize"] = tokenize_config_path
        seen["input"] = input_dir
        seen["output"] = output_dir
        seen["log_path"] = log_path
        return (Path("data/tokenized/tiny_stories/valid.jsonl"),)

    with patch("tiny_story.entrypoints.cli.tokenize.tokenize_dataset", fake_tokenize):
        result = runner.invoke(app, ["tokenize"])

    assert result.exit_code == 0
    assert seen["dataset"] == Path("config/dataset.yaml")
    assert seen["tokenize"] == Path("config/tokenize.yaml")
    assert seen["input"] is None
    assert seen["output"] is None
    assert seen["log_path"] is None
    assert "saved data/tokenized/tiny_stories/valid.jsonl" in result.output
    assert {download.__name__, tokenize.__name__} == {"download", "tokenize"}


def test_makefile_tokenize_runs_the_tokenize_command() -> None:
    """``make tokenize`` invokes the tokenize command."""
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "python -m tiny_story.entrypoints.cli tokenize" in makefile
