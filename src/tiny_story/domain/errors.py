"""Domain and download errors."""

from __future__ import annotations


class DomainError(Exception):
    """Failure of a domain rule or a dataset download."""


class InvalidDatasetSpecError(DomainError):
    """The dataset identity cannot be stored as raw files."""


class InvalidDatasetConfigError(DomainError):
    """The dataset configuration cannot be acquired."""


class InvalidRawFilenameError(DomainError):
    """A name is not a single file inside the raw directory."""


class DownloadError(DomainError):
    """A published file could not be retrieved."""


class DownloadUnavailableError(DownloadError):
    """The Hub could not be reached to retrieve a published file."""


class RawStorageError(DomainError):
    """A raw file could not be stored without changing its bytes."""


class InvalidCleanTextConfigError(DomainError):
    """The text cleaning configuration cannot be used."""


class CleanTextError(DomainError):
    """A text file could not be cleaned."""


class InvalidNormalizeTextConfigError(DomainError):
    """The text normalization configuration cannot be used."""


class NormalizeTextError(DomainError):
    """A text file could not be normalized."""
