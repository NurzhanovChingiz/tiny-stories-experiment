"""Path for one dataset download log."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class InvalidDatasetLogNameError(ValueError):
    """A dataset name is not a single log file name under ``download/logs``."""


def download_log_path(artifacts_root: Path, dataset_name: str) -> Path:
    """Return ``<artifacts_root>/download/logs/<dataset_name>.log``.

    Args:
        artifacts_root: Directory that contains ``download``.
        dataset_name: Single file stem, such as ``tiny_stories``.

    Returns:
        The download log file path.

    Raises:
        InvalidDatasetLogNameError: ``dataset_name`` is blank or not one file name.

    Flow:
        1. Reject a dataset name that is not a single file name.
        2. Place that name under download/logs.
    """
    # 1. Reject a dataset name that is not a single file name.
    name = _single_dataset_name(dataset_name)

    # 2. Place that name under download/logs.
    return artifacts_root / "download" / "logs" / f"{name}.log"


def _single_dataset_name(dataset_name: str) -> str:
    """Return a dataset name that names one log file.

    Args:
        dataset_name: Candidate log file stem.

    Returns:
        The same name when it is one file name.

    Raises:
        InvalidDatasetLogNameError: The name is blank or contains a path separator.
    """
    separator_present = (
        "/" in dataset_name or "\\" in dataset_name or "\x00" in dataset_name
    )
    blank = (
        dataset_name == ""
        or dataset_name.strip() != dataset_name
        or dataset_name in {".", ".."}
    )
    if blank or separator_present:
        message = "Dataset log name must be a single file name"
        raise InvalidDatasetLogNameError(message)
    return dataset_name
