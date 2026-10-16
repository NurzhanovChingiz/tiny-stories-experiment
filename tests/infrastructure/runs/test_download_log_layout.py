"""Behavior of download log artifact paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tiny_story.infrastructure.runs.download_log_layout import (
    InvalidDatasetLogNameError,
    clean_log_path,
    download_log_path,
    normalize_log_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    LogPathFor = Callable[[Path, str], Path]


@pytest.mark.parametrize(
    ("log_path_for", "stage"),
    [
        (download_log_path, "download"),
        (clean_log_path, "clean"),
        (normalize_log_path, "normalize"),
    ],
)
def test_stage_log_path_is_under_artifacts_stage_logs(
    tmp_path: Path,
    log_path_for: LogPathFor,
    stage: str,
) -> None:
    """A dataset name selects artifacts/<stage>/logs/<name>.log."""
    log_path = log_path_for(tmp_path / "artifacts", "tiny_stories")

    assert log_path == tmp_path / "artifacts" / stage / "logs" / "tiny_stories.log"


@pytest.mark.parametrize(
    "dataset_name",
    [
        "",
        " ",
        ".",
        "..",
        "tiny/stories",
        "tiny\\stories",
        "/tiny_stories",
        "tiny\x00stories",
    ],
)
@pytest.mark.parametrize(
    "log_path_for",
    [download_log_path, clean_log_path, normalize_log_path],
)
def test_stage_log_path_rejects_a_name_that_can_leave_the_logs_directory(
    tmp_path: Path,
    dataset_name: str,
    log_path_for: LogPathFor,
) -> None:
    """Empty and path-shaped names stay out of artifacts/<stage>/logs."""
    with pytest.raises(InvalidDatasetLogNameError, match="single file name"):
        log_path_for(tmp_path / "artifacts", dataset_name)
