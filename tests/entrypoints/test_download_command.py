"""Behavior of the download command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tiny_story.application.results.download_outcome import DownloadOutcome
from tiny_story.domain.errors import InvalidDatasetConfigError
from tiny_story.entrypoints.cli import app

runner = CliRunner()


def test_download_command_defaults_to_the_shipped_configuration() -> None:
    """The download command reads the shipped configuration unless flags override it."""
    seen: dict[str, Path | None] = {}

    def fake_acquire(
        config_path: Path,
        destination: Path | None = None,
        cache_dir: Path | None = None,
    ) -> DownloadOutcome:
        seen["config"] = config_path
        seen["destination"] = destination
        seen["cache"] = cache_dir
        return DownloadOutcome(
            directory=Path("data/raw/tiny_stories_raw"),
            written_paths=(),
            already_present_paths=(),
        )

    with patch("tiny_story.entrypoints.cli.acquire_dataset", fake_acquire):
        result = runner.invoke(app, ["download"])

    assert result.exit_code == 0
    assert seen["config"] == Path("config/dataset.yaml")
    assert seen["destination"] is None
    assert seen["cache"] is None


def test_download_command_passes_config_and_directory_overrides(tmp_path: Path) -> None:
    """Config, destination, and cache flags are forwarded to acquisition."""
    config = tmp_path / "dataset.yaml"
    raw = tmp_path / "raw"
    cache = tmp_path / "cache"
    seen: dict[str, Path | None] = {}

    def fake_acquire(
        config_path: Path,
        destination: Path | None = None,
        cache_dir: Path | None = None,
    ) -> DownloadOutcome:
        seen["config"] = config_path
        seen["destination"] = destination
        seen["cache"] = cache_dir
        return DownloadOutcome(
            directory=raw,
            written_paths=(raw / "valid.jsonl",),
            already_present_paths=(raw / "train.jsonl",),
        )

    with patch("tiny_story.entrypoints.cli.acquire_dataset", fake_acquire):
        result = runner.invoke(
            app,
            [
                "download",
                "--config",
                str(config),
                "--raw-dir",
                str(raw),
                "--cache-dir",
                str(cache),
            ],
        )

    assert result.exit_code == 0
    assert seen["config"] == config
    assert seen["destination"] == raw
    assert seen["cache"] == cache
    assert f"present {raw / 'train.jsonl'}" in result.output
    assert f"saved {raw / 'valid.jsonl'}" in result.output


def test_download_command_reports_a_missing_configuration(tmp_path: Path) -> None:
    """A missing configuration file fails the command."""
    missing = tmp_path / "missing.yaml"

    result = runner.invoke(app, ["download", "--config", str(missing)])

    assert result.exit_code != 0
    assert isinstance(result.exception, InvalidDatasetConfigError)
    assert "does not exist" in str(result.exception)


def test_download_command_reports_present_and_saved_files(tmp_path: Path) -> None:
    """The command prints files that were already stored and files just saved."""
    raw = tmp_path / "raw"
    present = raw / "train.jsonl"
    saved = raw / "valid.jsonl"

    def fake_acquire(
        config_path: Path,
        destination: Path | None = None,
        cache_dir: Path | None = None,
    ) -> DownloadOutcome:
        assert config_path == Path("config/dataset.yaml")
        assert destination is None
        assert cache_dir is None
        return DownloadOutcome(
            directory=raw,
            written_paths=(saved,),
            already_present_paths=(present,),
        )

    with patch("tiny_story.entrypoints.cli.acquire_dataset", fake_acquire):
        result = runner.invoke(app, ["download"])

    assert result.exit_code == 0
    assert f"present {present}" in result.output
    assert f"saved {saved}" in result.output
