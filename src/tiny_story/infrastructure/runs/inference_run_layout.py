"""Paths for one versioned inference run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class InvalidRunVersionError(ValueError):
    """A run version is not a single directory name under ``runs``."""


@dataclass(frozen=True, slots=True)
class InferenceRunLayout:
    """Log file and JSON prediction directory for one inference run.

    Attributes:
        log_path: ``runs/<version>/logs/inference.log`` under the artifacts root.
        predictions_dir: ``runs/<version>/predictions/json`` under the artifacts root.
    """

    log_path: Path
    predictions_dir: Path


def inference_run_layout(artifacts_root: Path, version: str) -> InferenceRunLayout:
    """Return the inference log and JSON prediction directory for one version.

    The run directory is ``<artifacts_root>/runs/<version>``. The log file is
    ``logs/inference.log``. JSON predictions are written under
    ``predictions/json``.

    Args:
        artifacts_root: Directory that contains ``runs``.
        version: Single directory name, such as ``version_n``.

    Returns:
        The log file path and the JSON predictions directory.

    Raises:
        InvalidRunVersionError: ``version`` is blank or not one directory name.

    Flow:
        1. Reject a version that is not a single directory name.
        2. Place the inference log and JSON predictions under that run.
    """
    # 1. Reject a version that is not a single directory name.
    run_dir = artifacts_root / "runs" / _single_version(version)

    # 2. Place the inference log and JSON predictions under that run.
    return InferenceRunLayout(
        log_path=run_dir / "logs" / "inference.log",
        predictions_dir=run_dir / "predictions" / "json",
    )


def _single_version(version: str) -> str:
    """Return a version that names one directory.

    Args:
        version: Candidate run directory name.

    Returns:
        The same version when it is one directory name.

    Raises:
        InvalidRunVersionError: The version is blank or contains a path separator.
    """
    separator_present = "/" in version or "\\" in version or "\x00" in version
    blank = version == "" or version.strip() != version or version in {".", ".."}
    if blank or separator_present:
        message = "Run version must be a single directory name"
        raise InvalidRunVersionError(message)
    return version
