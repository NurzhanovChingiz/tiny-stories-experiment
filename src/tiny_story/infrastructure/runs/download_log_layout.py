"""Paths for dataset run logs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class InvalidDatasetLogNameError(ValueError):
    """A dataset name is not a single log file name."""


def download_log_path(artifacts_root: Path, dataset_name: str) -> Path:
    """Return ``<artifacts_root>/download/logs/<dataset_name>.log``.

    Args:
        artifacts_root: Directory that contains ``download``.
        dataset_name: Single file stem, such as ``tiny_stories``.

    Returns:
        The download log file path.

    Raises:
        InvalidDatasetLogNameError: ``dataset_name`` is blank or not one file name.
    """
    return _stage_log_path(artifacts_root, "download", dataset_name)


def clean_log_path(artifacts_root: Path, dataset_name: str) -> Path:
    """Return ``<artifacts_root>/clean/logs/<dataset_name>.log``.

    Args:
        artifacts_root: Directory that contains ``clean``.
        dataset_name: Single file stem, such as ``tiny_stories``.

    Returns:
        The cleaning log file path.

    Raises:
        InvalidDatasetLogNameError: ``dataset_name`` is blank or not one file name.
    """
    return _stage_log_path(artifacts_root, "clean", dataset_name)


def normalize_log_path(artifacts_root: Path, dataset_name: str) -> Path:
    """Return ``<artifacts_root>/normalize/logs/<dataset_name>.log``.

    Args:
        artifacts_root: Directory that contains ``normalize``.
        dataset_name: Single file stem, such as ``tiny_stories``.

    Returns:
        The normalization log file path.

    Raises:
        InvalidDatasetLogNameError: ``dataset_name`` is blank or not one file name.
    """
    return _stage_log_path(artifacts_root, "normalize", dataset_name)


def _stage_log_path(artifacts_root: Path, stage: str, dataset_name: str) -> Path:
    """Return ``<artifacts_root>/<stage>/logs/<dataset_name>.log``.

    Args:
        artifacts_root: Directory that contains the stage directory.
        stage: Single path segment, such as ``download``.
        dataset_name: Single file stem, such as ``tiny_stories``.

    Returns:
        The stage log file path.

    Raises:
        InvalidDatasetLogNameError: ``dataset_name`` is blank or not one file name.

    Flow:
        1. Reject a dataset name that is not a single file name.
        2. Place that name under the stage logs directory.
    """
    # 1. Reject a dataset name that is not a single file name.
    name = _single_dataset_name(dataset_name)

    # 2. Place that name under the stage logs directory.
    return artifacts_root / stage / "logs" / f"{name}.log"


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
