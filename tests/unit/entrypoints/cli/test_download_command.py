"""CLI progress lines for the TinyStoriesV2 download."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from tiny_stories_experiment.application.results.download_outcome import DownloadOutcome
from tiny_stories_experiment.composition import TINY_STORIES_V2
from tiny_stories_experiment.entrypoints.cli.commands.download import download_command

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def restored_loguru() -> Iterator[None]:
    """Put the default stderr sink back after the command replaces it."""
    yield
    logger.remove()
    logger.add(sys.stderr)


def test_download_command_logs_kept_and_downloaded_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    restored_loguru: None,
) -> None:
    """Progress lines name the dataset, kept files, downloads, and raw directory."""
    del restored_loguru
    destination = tmp_path / "raw"
    outcome = DownloadOutcome(
        destination=destination,
        kept=("TinyStoriesV2-GPT4-train.jsonl",),
        downloaded=("TinyStoriesV2-GPT4-valid.jsonl",),
    )

    def fake_download(raw_directory: Path) -> DownloadOutcome:
        assert raw_directory == destination
        return outcome

    monkeypatch.setattr(
        "tiny_stories_experiment.entrypoints.cli.commands.download.download_default_dataset",
        fake_download,
    )

    download_command(destination)
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err.splitlines() == [
        f"dataset {TINY_STORIES_V2.source_url}",
        "kept TinyStoriesV2-GPT4-train.jsonl",
        "downloaded TinyStoriesV2-GPT4-valid.jsonl",
        f"raw directory {destination}",
    ]
