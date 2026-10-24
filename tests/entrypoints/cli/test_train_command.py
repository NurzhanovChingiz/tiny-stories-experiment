"""Behavior of the train command and its Makefile target."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tiny_story.application.results.train_outcome import TrainOutcome
from tiny_story.entrypoints.cli.app import app
from tiny_story.entrypoints.cli.train import train

runner = CliRunner()


def test_train_command_uses_the_shipped_configuration() -> None:
    """The train command reads the shipped configuration unless flags override it."""
    seen: dict[str, Path | None] = {}

    def fake_train(
        config_path: Path,
        data_dir: Path | None = None,
        checkpoint_dir: Path | None = None,
        log_dir: Path | None = None,
        log_path: Path | None = None,
    ) -> TrainOutcome:
        seen["config"] = config_path
        seen["data"] = data_dir
        seen["checkpoint"] = checkpoint_dir
        seen["log_dir"] = log_dir
        seen["log_path"] = log_path
        return TrainOutcome(
            checkpoint_dir=Path("artifacts/train/checkpoints"),
            log_dir=Path("artifacts/train/lightning_logs"),
            best_model_path=Path("artifacts/train/checkpoints/best.ckpt"),
        )

    with patch("tiny_story.entrypoints.cli.train.train_model", fake_train):
        result = runner.invoke(app, ["train"])

    assert result.exit_code == 0
    assert seen["config"] == Path("config/train.yaml")
    assert seen["data"] is None
    assert seen["checkpoint"] is None
    assert seen["log_dir"] is None
    assert seen["log_path"] is None
    assert "saved artifacts/train/checkpoints/best.ckpt" in result.output
    assert train.__name__ == "train"


def test_makefile_train_runs_the_train_command() -> None:
    """``make train`` invokes the train command."""
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "python -m tiny_story.entrypoints.cli train" in makefile
