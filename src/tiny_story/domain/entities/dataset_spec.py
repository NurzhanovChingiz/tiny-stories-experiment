"""Identity of a raw dataset."""

from __future__ import annotations

from dataclasses import dataclass

from tiny_story.domain.errors import InvalidDatasetSpecError, InvalidRawFilenameError


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Identity of a dataset stored as raw files.

    Attributes:
        name: Local dataset name.
        filenames: Raw file names to place in storage.
    """

    name: str
    filenames: tuple[str, ...]


def require_raw_filename(filename: str) -> str:
    """Return a filename that stays inside the raw directory.

    Args:
        filename: Candidate raw file name.

    Returns:
        The same filename when it names one file and nothing outside that directory.

    Raises:
        InvalidRawFilenameError: The name is empty, absolute, or contains a path separator.
    """
    separator_present = "/" in filename or "\\" in filename or "\x00" in filename
    blank = filename == "" or filename.strip() != filename or filename in {".", ".."}
    if blank or separator_present:
        message = "Raw filename must be a single file name"
        raise InvalidRawFilenameError(message)
    return filename


def require_dataset_spec(spec: DatasetSpec) -> DatasetSpec:
    """Return the spec when its name and filenames can be stored raw.

    Args:
        spec: Dataset identity.

    Returns:
        The same spec when the name is usable and every filename is one raw file name.

    Raises:
        InvalidDatasetSpecError: The name is blank or a file name is missing or repeated.
        InvalidRawFilenameError: A filename would leave the raw directory.

    Flow:
        1. Reject a blank dataset name.
        2. Reject missing, repeated, or unsafe file names.
    """
    # 1. Reject a blank dataset name.
    if spec.name == "" or spec.name.strip() != spec.name:
        message = "Dataset name must be a non-empty string"
        raise InvalidDatasetSpecError(message)

    # 2. Reject missing, repeated, or unsafe file names.
    if len(spec.filenames) == 0 or len(set(spec.filenames)) != len(spec.filenames):
        message = "Dataset spec must name each raw file once"
        raise InvalidDatasetSpecError(message)
    for filename in spec.filenames:
        require_raw_filename(filename)
    return spec
