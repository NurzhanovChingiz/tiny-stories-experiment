"""Domain errors for dataset download and preparation."""


class RawFileConflictError(Exception):
    """Local raw file size differs from the published file."""


class DownloadSizeMismatchError(Exception):
    """Fetched file size differs from the published byte count."""


class PublishedSizeMissingError(Exception):
    """Hub metadata for a published file did not include a byte count."""


class RawDatasetMissingError(Exception):
    """A raw file named by the dataset spec is absent."""


class UnknownDatasetSplitError(Exception):
    """A raw file name does not end in a published split token."""


class RawDatasetChangedError(Exception):
    """A raw file's size changed while derived text was written."""


class RawStoryRecordError(Exception):
    """A raw JSONL line is not an object with a string text field."""
