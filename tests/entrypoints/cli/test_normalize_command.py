"""Behavior of the normalize command and its Makefile target."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tiny_story.entrypoints.cli.app import app
from tiny_story.entrypoints.cli.download import download
from tiny_story.entrypoints.cli.normalize import normalize

runner = CliRunner()


def test_normalize_command_uses_the_shipped_configuration() -> None:
    """The normalize command reads the shipped configuration unless flags override it."""
    seen: dict[str, Path | None] = {}

    def fake_normalize(
        dataset_config_path: Path,
        normalize_config_path: Path,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        log_path: Path | None = None,
    ) -> tuple[Path, ...]:
        seen["dataset"] = dataset_config_path
        seen["normalize"] = normalize_config_path
        seen["input"] = input_dir
        seen["output"] = output_dir
        seen["log_path"] = log_path
        return (Path("data/processed/tiny_stories/valid.jsonl"),)

    with patch(
        "tiny_story.entrypoints.cli.normalize.normalize_dataset", fake_normalize
    ):
        result = runner.invoke(app, ["normalize"])

    assert result.exit_code == 0
    assert seen["dataset"] == Path("config/dataset.yaml")
    assert seen["normalize"] == Path("config/normalize_text.yaml")
    assert seen["input"] is None
    assert seen["output"] is None
    assert seen["log_path"] is None
    assert "saved data/processed/tiny_stories/valid.jsonl" in result.output
    assert {download.__name__, normalize.__name__} == {"download", "normalize"}


def test_normalize_command_passes_an_explicit_log_path(tmp_path: Path) -> None:
    """The log-path flag is forwarded to normalization."""
    log_path = tmp_path / "run.log"
    seen: dict[str, Path | None] = {}

    def fake_normalize(
        dataset_config_path: Path,
        normalize_config_path: Path,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        log_path: Path | None = None,
    ) -> tuple[Path, ...]:
        del dataset_config_path, normalize_config_path, input_dir, output_dir
        seen["log_path"] = log_path
        return ()

    with patch(
        "tiny_story.entrypoints.cli.normalize.normalize_dataset", fake_normalize
    ):
        result = runner.invoke(app, ["normalize", "--log-path", str(log_path)])

    assert result.exit_code == 0
    assert seen["log_path"] == log_path


def test_makefile_clean_and_normalize_run_separate_commands() -> None:
    """``make clean`` and ``make normalize`` invoke different commands."""
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "python -m tiny_story.entrypoints.cli clean" in makefile
    assert "python -m tiny_story.entrypoints.cli normalize" in makefile
