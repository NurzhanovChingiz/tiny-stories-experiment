"""Behavior of the clean command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tiny_story.entrypoints.cli.app import app
from tiny_story.entrypoints.cli.clean import clean
from tiny_story.entrypoints.cli.download import download

runner = CliRunner()


def test_clean_command_forwards_the_log_path(tmp_path: Path) -> None:
    """The clean command uses the default log path unless --log-path is set."""
    log_path = tmp_path / "run.log"
    seen: dict[str, Path | None] = {}
    log_paths: list[Path | None] = []

    def fake_clean(
        dataset_config_path: Path,
        clean_config_path: Path,
        output_dir: Path | None = None,
        log_path: Path | None = None,
    ) -> tuple[Path, ...]:
        seen["dataset"] = dataset_config_path
        seen["clean"] = clean_config_path
        seen["output"] = output_dir
        log_paths.append(log_path)
        return (Path("data/interim/tiny_stories/valid.jsonl"),)

    with patch("tiny_story.entrypoints.cli.clean.clean_dataset", fake_clean):
        default = runner.invoke(app, ["clean"])
        override = runner.invoke(app, ["clean", "--log-path", str(log_path)])

    assert {clean.__name__, download.__name__} == {"clean", "download"}
    assert default.exit_code == 0
    assert seen["dataset"] == Path("config/dataset.yaml")
    assert seen["clean"] == Path("config/clean_text.yaml")
    assert seen["output"] is None
    assert "saved data/interim/tiny_stories/valid.jsonl" in default.output
    assert override.exit_code == 0
    assert log_paths == [None, log_path]
