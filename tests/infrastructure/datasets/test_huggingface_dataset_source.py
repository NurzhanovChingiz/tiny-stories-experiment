"""Behavior of the Hugging Face dataset adapter."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx2
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    XetDownloadError,
)

from tiny_story.domain.errors import DownloadError, DownloadUnavailableError
from tiny_story.infrastructure.datasets.huggingface_dataset_source import (
    HuggingFaceDatasetSource,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("repository", "revision", "filename"),
    [
        ("noanabeshima/TinyStoriesV2", "main", "TinyStoriesV2-GPT4-train.jsonl"),
        ("some-org/other-corpus", "abc123", "notes.txt"),
    ],
)
def test_fetch_returns_the_cached_file_for_any_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository: str,
    revision: str,
    filename: str,
) -> None:
    """The adapter requests the configured repository, revision, and file."""
    cached = tmp_path / "cache" / filename
    cached.parent.mkdir()
    cached.write_bytes(b"story\n")
    seen: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        seen.update(kwargs)
        return str(cached)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )

    fetched = HuggingFaceDatasetSource(tmp_path / "cache", repository, revision).fetch(
        filename,
    )

    assert fetched == cached
    assert cached.read_bytes() == b"story\n"
    assert seen == {
        "repo_id": repository,
        "filename": filename,
        "repo_type": "dataset",
        "revision": revision,
        "cache_dir": tmp_path / "cache",
    }


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (EntryNotFoundError("missing"), DownloadError),
        (HFValidationError("bad"), DownloadError),
        (LocalEntryNotFoundError("missing"), DownloadError),
        (OSError("disk"), DownloadError),
        (XetDownloadError("xet"), DownloadUnavailableError),
        (httpx2.ConnectError("connect"), DownloadUnavailableError),
        (httpx2.TimeoutException("timeout"), DownloadUnavailableError),
        (httpx2.ProtocolError("protocol"), DownloadUnavailableError),
        (httpx2.ProxyError("proxy"), DownloadUnavailableError),
    ],
)
def test_fetch_maps_hub_failures_to_download_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_type: type[DownloadError],
) -> None:
    """Hub and filesystem failures become a download error."""

    def fake_download(**kwargs: object) -> str:
        assert kwargs["filename"] == "notes.txt"
        raise error

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )

    with pytest.raises(
        expected_type,
        match=r"Could not download notes\.txt from some-org/other-corpus",
    ) as caught:
        HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "main").fetch(
            "notes.txt",
        )

    assert type(caught.value) is expected_type
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("operation", ["fetch", "content_sha256"])
def test_xet_download_error_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """A Xet download failure stays unavailable so a local copy can be kept."""
    error = XetDownloadError("xet")

    def fake_download(**kwargs: object) -> str:
        assert kwargs["filename"] == "notes.txt"
        raise error

    def fake_metadata(_url: str, **_kwargs: object) -> SimpleNamespace:
        raise error

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.get_hf_file_metadata",
        fake_metadata,
    )
    source = HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "main")

    with pytest.raises(
        DownloadUnavailableError,
        match=r"Could not download notes\.txt from some-org/other-corpus",
    ) as caught:
        getattr(source, operation)("notes.txt")

    assert type(caught.value) is DownloadUnavailableError
    assert caught.value.__cause__ is error


def test_fetch_rejects_a_download_that_does_not_return_a_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-path Hub result is a download error."""

    def fake_download(**kwargs: object) -> object:
        assert kwargs["repo_id"] == "some-org/other-corpus"
        return object()

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )

    with pytest.raises(
        DownloadError,
        match=r"Could not download notes\.txt from some-org/other-corpus",
    ):
        HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "main").fetch(
            "notes.txt",
        )


def test_publication_id_names_the_repository_and_revision(tmp_path: Path) -> None:
    """The publication id is a stable JSON array of repository and revision."""
    source = HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "abc123")
    other = HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "def456")

    assert source.publication_id == '["some-org/other-corpus","abc123"]'
    assert source.publication_id != other.publication_id


def test_content_sha256_returns_a_quoted_etag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quoted Hub etag is the published SHA-256."""
    digest = "a" * 64

    def fake_metadata(url: str, **_kwargs: object) -> SimpleNamespace:
        assert url.endswith("notes.txt")
        return SimpleNamespace(etag=f'W/"{digest.upper()}"')

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.get_hf_file_metadata",
        fake_metadata,
    )

    actual = HuggingFaceDatasetSource(
        tmp_path,
        "some-org/other-corpus",
        "abc123",
    ).content_sha256("notes.txt")

    assert actual == digest


def test_fetch_rejects_bytes_that_differ_from_the_published_etag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A published SHA-256 is not the digest of different downloaded bytes."""
    published = "ab" * 32
    payload = b"notes\n"
    cached = tmp_path / "notes.txt"
    cached.write_bytes(payload)
    calls: list[str] = []

    def fake_metadata(_url: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(etag=published)

    def fake_download(**kwargs: object) -> str:
        calls.append(str(kwargs["filename"]))
        return str(cached)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.get_hf_file_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    source = HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "main")

    digest = source.content_sha256("notes.txt")
    with pytest.raises(
        DownloadError,
        match=r"Could not download notes\.txt from some-org/other-corpus",
    ) as caught:
        source.fetch("notes.txt")

    assert digest == published
    assert digest != hashlib.sha256(payload).hexdigest()
    assert type(caught.value) is DownloadError
    assert calls == ["notes.txt"]


def test_fetch_returns_bytes_that_match_the_published_etag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A published SHA-256 is returned with the file whose bytes hash to it."""
    payload = b"notes\n"
    published = hashlib.sha256(payload).hexdigest()
    cached = tmp_path / "notes.txt"
    cached.write_bytes(payload)

    def fake_metadata(_url: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(etag=published)

    def fake_download(**kwargs: object) -> str:
        assert kwargs["filename"] == "notes.txt"
        return str(cached)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.get_hf_file_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    source = HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "main")

    digest = source.content_sha256("notes.txt")
    fetched = source.fetch("notes.txt")

    assert digest == published
    assert fetched == cached
    assert hashlib.sha256(fetched.read_bytes()).hexdigest() == digest


def test_fetch_reuses_the_file_hashed_for_a_non_sha256_etag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hashing a non-SHA-256 etag downloads the file once for the following fetch."""
    cached = tmp_path / "notes.txt"
    cached.write_bytes(b"notes\n")
    calls: list[str] = []

    def fake_metadata(_url: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(etag="cd60da109be2fcdced226d1ace068799232cb97a")

    def fake_download(**kwargs: object) -> str:
        calls.append(str(kwargs["filename"]))
        return str(cached)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.get_hf_file_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    source = HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "main")

    digest = source.content_sha256("notes.txt")
    fetched = source.fetch("notes.txt")

    assert digest == hashlib.sha256(b"notes\n").hexdigest()
    assert fetched == cached
    assert calls == ["notes.txt"]


def test_fetch_reports_offline_mode_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub offline mode is an unavailable download, so a local copy can be kept."""
    error = OfflineModeIsEnabled("offline")

    def fake_download(**_kwargs: object) -> str:
        raise error

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )

    with pytest.raises(
        DownloadUnavailableError,
        match=r"Could not download notes\.txt from some-org/other-corpus",
    ) as caught:
        HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "main").fetch(
            "notes.txt",
        )

    assert type(caught.value) is DownloadUnavailableError
    assert caught.value.__cause__ is error


def test_content_sha256_reports_a_retryable_status_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retryable Hub status is an unavailable download."""
    error = HfHubHTTPError(
        "unavailable",
        response=httpx2.Response(
            503,
            request=httpx2.Request("GET", "https://example.test/notes.txt"),
        ),
    )

    def fake_metadata(_url: str, **_kwargs: object) -> SimpleNamespace:
        raise error

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.get_hf_file_metadata",
        fake_metadata,
    )

    with pytest.raises(DownloadUnavailableError) as caught:
        HuggingFaceDatasetSource(
            tmp_path,
            "some-org/other-corpus",
            "main",
        ).content_sha256("notes.txt")

    assert type(caught.value) is DownloadUnavailableError
