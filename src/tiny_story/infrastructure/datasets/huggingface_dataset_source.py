"""Hugging Face adapter for one configured dataset."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx2
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    XetDownloadError,
)

from tiny_story.application.ports.dataset_source import DatasetSource
from tiny_story.domain.errors import DownloadError, DownloadUnavailableError

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_DOWNLOAD_FAILURES = (
    EntryNotFoundError,
    GatedRepoError,
    HFValidationError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OSError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    XetDownloadError,
    httpx2.TransportError,
)


class HuggingFaceDatasetSource(DatasetSource):
    """Downloads one file from a configured Hugging Face dataset repository."""

    def __init__(self, cache_dir: Path, repository: str, revision: str) -> None:
        """Store the repository, revision, and interim cache.

        Args:
            cache_dir: Directory for the Hugging Face cache.
            repository: Hugging Face dataset id, such as ``owner/name``.
            revision: Dataset revision, branch, tag, or commit.

        Returns:
            None.
        """
        self._cache_dir = cache_dir
        self._repository = repository
        self._revision = revision
        self._hashed_downloads: dict[str, Path] = {}
        self._observed_sha256: dict[str, str] = {}

    @property
    def publication_id(self) -> str:
        """Return the repository and revision this adapter downloads.

        Returns:
            A JSON array of the repository id and the revision.
        """
        return json.dumps(
            [self._repository, self._revision],
            separators=(",", ":"),
        )

    def fetch(self, filename: str) -> Path:
        """Download one dataset file and return its cache path.

        Args:
            filename: File name inside the configured repository.

        Returns:
            Path of the downloaded file. The file is not modified.

        Raises:
            DownloadError: The file could not be retrieved, or its bytes do
                not match a SHA-256 previously returned for this name.
            DownloadUnavailableError: The Hub could not be reached.

        Flow:
            1. Return a file already hashed for this name.
            2. Download the file when it has not been hashed yet.
            3. Return that file when no SHA-256 was remembered or the file matches it.
        """
        # 1. Return a file already hashed for this name.
        cached = self._take_hashed_download(filename)
        if cached is not None:
            return cached

        # 2. Download the file when it has not been hashed yet.
        downloaded = self._download(filename)

        # 3. Return that file when no SHA-256 was remembered or the file matches it.
        self._require_matching_sha256(filename, downloaded)
        return downloaded

    def content_sha256(self, filename: str) -> str:
        """Return the SHA-256 of one dataset file.

        Args:
            filename: File name inside the configured repository.

        Returns:
            Lowercase SHA-256 hex digest of that file.

        Raises:
            DownloadError: The digest could not be retrieved.
            DownloadUnavailableError: The Hub could not be reached.

        Flow:
            1. Return the SHA-256 carried by the Hub etag when it is one, and remember it for the next fetch.
            2. Hash the downloaded file when the etag is not a SHA-256, and keep that file for the next fetch.
        """
        # 1. Return the SHA-256 carried by the Hub etag when it is one, and remember it for the next fetch.
        published = self._published_sha256(filename)
        if published is not None:
            self._observed_sha256[filename] = published
            return published

        # 2. Hash the downloaded file when the etag is not a SHA-256, and keep that file for the next fetch.
        return self._sha256_from_download(filename)

    def _take_hashed_download(self, filename: str) -> Path | None:
        """Return the file just hashed for ``filename`` when it is still present.

        Args:
            filename: Raw file name.

        Returns:
            The cached download, or None when this name was not hashed yet
            or that file is gone.
        """
        downloaded = self._hashed_downloads.pop(filename, None)
        if downloaded is not None and downloaded.is_file():
            return downloaded
        return None

    def _require_matching_sha256(self, filename: str, downloaded: Path) -> None:
        """Reject a download whose bytes are not the remembered SHA-256.

        Args:
            filename: Raw file name.
            downloaded: File just retrieved for that name.

        Returns:
            None when this name has no remembered SHA-256 or the file matches it.

        Raises:
            DownloadError: The file hash differs from the remembered SHA-256,
                or the file could not be hashed.

        Flow:
            1. Skip the check when this name has no remembered SHA-256.
            2. Hash the file and reject it when the digest differs.
        """
        # 1. Skip the check when this name has no remembered SHA-256.
        expected = self._observed_sha256.get(filename)
        if expected is None:
            return

        # 2. Hash the file and reject it when the digest differs.
        try:
            actual = _sha256_hex(downloaded)
        except OSError as exc:
            failure = DownloadError(
                _download_failure_message(filename, self._repository)
            )
            raise failure from exc
        if actual != expected:
            raise DownloadError(_download_failure_message(filename, self._repository))

    def _sha256_from_download(self, filename: str) -> str:
        """Hash a file whose Hub etag is not a SHA-256 and keep that file.

        Args:
            filename: File name inside the configured repository.

        Returns:
            Lowercase SHA-256 hex digest of the downloaded file.

        Raises:
            DownloadError: The file could not be retrieved or hashed.
            DownloadUnavailableError: The Hub could not be reached.

        Flow:
            1. Download the file.
            2. Hash that file and keep it for the next fetch.
        """
        # 1. Download the file.
        downloaded = self._download(filename)

        # 2. Hash that file and keep it for the next fetch.
        try:
            digest = _sha256_hex(downloaded)
        except OSError as exc:
            failure = DownloadError(
                _download_failure_message(filename, self._repository)
            )
            raise failure from exc
        self._hashed_downloads[filename] = downloaded
        return digest

    def _download(self, filename: str) -> Path:
        """Download one dataset file and return its cache path.

        Args:
            filename: File name inside the configured repository.

        Returns:
            Path of the downloaded file. The file is not modified.

        Raises:
            DownloadError: The file could not be retrieved.
            DownloadUnavailableError: The Hub could not be reached.

        Flow:
            1. Download the requested file into the adapter cache.
            2. Return the cache path when the download produced a file.
        """
        # 1. Download the requested file into the adapter cache.
        try:
            downloaded = hf_hub_download(
                repo_id=self._repository,
                filename=filename,
                repo_type="dataset",
                revision=self._revision,
                cache_dir=self._cache_dir,
            )
        except _DOWNLOAD_FAILURES as exc:
            raise self._download_error(filename, exc) from exc

        # 2. Return the cache path when the download produced a file.
        if isinstance(downloaded, str):
            return Path(downloaded)
        raise DownloadError(_download_failure_message(filename, self._repository))

    def _published_sha256(self, filename: str) -> str | None:
        """Return the Hub etag when it is a SHA-256.

        Args:
            filename: File name inside the configured repository.

        Returns:
            The lowercase SHA-256, or None when the etag is not one.

        Raises:
            DownloadError: Hub metadata could not be retrieved.
            DownloadUnavailableError: The Hub could not be reached.

        Flow:
            1. Request Hub metadata for the file at the configured revision.
            2. Return the SHA-256 carried by the Hub etag.
        """
        # 1. Request Hub metadata for the file at the configured revision.
        try:
            metadata = get_hf_file_metadata(
                hf_hub_url(
                    repo_id=self._repository,
                    filename=filename,
                    repo_type="dataset",
                    revision=self._revision,
                ),
                retry_on_errors=False,
            )
        except _DOWNLOAD_FAILURES as exc:
            raise self._download_error(filename, exc) from exc

        # 2. Return the SHA-256 carried by the Hub etag.
        return _sha256_etag(metadata.etag)

    def _download_error(self, filename: str, exc: Exception) -> DownloadError:
        """Classify one Hub failure as unreachable or permanent.

        Args:
            filename: Raw file name that could not be retrieved.
            exc: Failure raised by the Hub client.

        Returns:
            An unavailable error when the Hub could not be reached, otherwise
            a permanent download error.

        Flow:
            1. Report a transport failure, a Xet download failure, Hub offline mode, or a retryable HTTP status as unreachable.
            2. Report every other Hub failure as a permanent download error.
        """
        # 1. Report a transport failure, a Xet download failure, Hub offline mode, or a retryable HTTP status as unreachable.
        message = _download_failure_message(filename, self._repository)
        if _hub_unreachable(exc):
            return DownloadUnavailableError(message)

        # 2. Report every other Hub failure as a permanent download error.
        return DownloadError(message)


def _hub_unreachable(exc: Exception) -> bool:
    """Return whether ``exc`` means the Hub could not be reached.

    Args:
        exc: Failure raised by the Hub client.

    Returns:
        True for a transport failure, a Xet download failure, Hub offline
        mode, or a retryable HTTP status.
    """
    return isinstance(
        exc,
        (httpx2.TransportError, OfflineModeIsEnabled, XetDownloadError),
    ) or _retryable_status(exc)


def _retryable_status(exc: Exception) -> bool:
    """Return whether ``exc`` carries a retryable HTTP status.

    Args:
        exc: Failure raised by the Hub client.

    Returns:
        True when the response status is 408, 429, 500, 502, 503, or 504.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return isinstance(status, int) and status in _RETRYABLE_STATUS


def _download_failure_message(filename: str, repository: str) -> str:
    """Return the download failure message on one line.

    Args:
        filename: Raw file name that could not be retrieved.
        repository: Hugging Face dataset id.

    Returns:
        ``Could not download {filename} from {repository}``.
    """
    return f"Could not download {filename} from {repository}"


def _sha256_etag(etag: str | None) -> str | None:
    """Return a lowercase SHA-256 when ``etag`` is that digest.

    Args:
        etag: Hub ETag, possibly quoted, or None when the Hub omitted it.

    Returns:
        The 64-character digest, or None when the etag is not a SHA-256.
    """
    if etag is None:
        return None
    normalized = etag.removeprefix("W/").strip('"').lower()
    if _SHA256_HEX.fullmatch(normalized) is None:
        return None
    return normalized


def _sha256_hex(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file.

    Args:
        path: Regular file to hash.

    Returns:
        The SHA-256 digest of the file bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()
