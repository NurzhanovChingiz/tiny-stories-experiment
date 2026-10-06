"""Domain errors for dataset download."""


class RawFileConflictError(Exception):
    """Local raw file size differs from the published file."""


class DownloadSizeMismatchError(Exception):
    """Fetched file size differs from the published byte count."""


class PublishedSizeMissingError(Exception):
    """Hub metadata for a published file did not include a byte count."""
