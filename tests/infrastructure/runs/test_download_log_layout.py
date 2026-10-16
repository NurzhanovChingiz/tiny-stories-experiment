"""Behavior of download log artifact paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tiny_story.infrastructure.runs.download_log_layout import (
    InvalidDatasetLogNameError,
    download_log_path,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_download_log_path_is_under_artifacts_download_logs(tmp_path: Path) -> None:
    """A dataset name selects artifacts/download/logs/<name>.log."""
    log_path = download_log_path(tmp_path / "artifacts", "tiny_stories")

    assert log_path == (
        tmp_path / "artifacts" / "download" / "logs" / "tiny_stories.log"
    )


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
def test_download_log_path_rejects_a_name_that_can_leave_download_logs(
    tmp_path: Path,
    dataset_name: str,
) -> None:
    """Empty and path-shaped names stay out of artifacts/download/logs."""
    with pytest.raises(InvalidDatasetLogNameError, match="single file name"):
        download_log_path(tmp_path / "artifacts", dataset_name)
