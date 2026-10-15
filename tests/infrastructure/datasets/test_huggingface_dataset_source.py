"""Behavior of the Hugging Face dataset adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx2
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    HFValidationError,
    LocalEntryNotFoundError,
    XetDownloadError,
)

from tiny_story.domain.errors import DownloadError
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
    "error",
    [
        EntryNotFoundError("missing"),
        HFValidationError("bad"),
        LocalEntryNotFoundError("missing"),
        OSError("disk"),
        XetDownloadError("xet"),
        httpx2.ConnectError("connect"),
        httpx2.TimeoutException("timeout"),
        httpx2.ProtocolError("protocol"),
        httpx2.ProxyError("proxy"),
    ],
)
def test_fetch_maps_hub_failures_to_download_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
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
        DownloadError,
        match=r"Could not download notes\.txt from some-org/other-corpus",
    ) as caught:
        HuggingFaceDatasetSource(tmp_path, "some-org/other-corpus", "main").fetch(
            "notes.txt",
        )

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
