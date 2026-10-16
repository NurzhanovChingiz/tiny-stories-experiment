"""Behavior of versioned inference artifact paths."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tiny_story.infrastructure.logging.filesystem_run_logger import FilesystemRunLogger
from tiny_story.infrastructure.runs.inference_run_layout import (
    InferenceRunLayout,
    InvalidRunVersionError,
    inference_run_layout,
)
from tiny_story.infrastructure.writers.filesystem_prediction_writer import (
    FilesystemPredictionWriter,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_inference_run_layout_matches_the_versioned_artifact_tree(
    tmp_path: Path,
) -> None:
    """A version names logs/inference.log and predictions/json under artifacts/runs."""
    layout = inference_run_layout(tmp_path / "artifacts", "version_n")

    assert layout == InferenceRunLayout(
        log_path=tmp_path
        / "artifacts"
        / "runs"
        / "version_n"
        / "logs"
        / "inference.log",
        predictions_dir=(
            tmp_path / "artifacts" / "runs" / "version_n" / "predictions" / "json"
        ),
    )


def test_logger_and_writer_store_inference_outputs_in_that_tree(
    tmp_path: Path,
) -> None:
    """Logging and JSON prediction writing create the versioned artifact files."""
    layout = inference_run_layout(tmp_path / "artifacts", "version_n")
    logger = FilesystemRunLogger(layout.log_path)
    try:
        logger.log("scored one story")
    finally:
        logger.close()
    FilesystemPredictionWriter(layout.predictions_dir).write(
        "story-1",
        {"label": "ok", "score": 1},
    )

    assert "scored one story" in layout.log_path.read_text(encoding="utf-8")
    stored = json.loads(
        (layout.predictions_dir / "story-1.json").read_text(encoding="utf-8")
    )
    assert stored == {"label": "ok", "score": 1}


@pytest.mark.parametrize(
    "version",
    ["", " ", ".", "..", "version/n", "version\\n", "/version_n", "version\x00n"],
)
def test_inference_run_layout_rejects_a_version_that_can_leave_runs(
    tmp_path: Path,
    version: str,
) -> None:
    """Empty and path-shaped versions stay out of artifacts/runs."""
    with pytest.raises(InvalidRunVersionError, match="single directory name"):
        inference_run_layout(tmp_path / "artifacts", version)
