"""Adapter tests for the Hugging Face TinyStories source."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tiny_stories_experiment.domain.errors import PublishedSizeMissingError
from tiny_stories_experiment.infrastructure.datasets import (
    huggingface_tinystories_repository as hub_repo,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_byte_count_reads_hub_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter returns the size from Hub file metadata."""

    class Metadata:
        size = 5

    monkeypatch.setattr(
        hub_repo, "hf_hub_url", lambda **_kwargs: "https://example.test/file"
    )
    monkeypatch.setattr(hub_repo, "get_hf_file_metadata", lambda _url: Metadata())
    source = hub_repo.HuggingfaceTinystoriesRepository("noanabeshima/TinyStoriesV2")
    assert source.byte_count("train.jsonl") == 5


def test_missing_hub_size_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Hub size metadata becomes a domain error."""

    class Metadata:
        size = None

    monkeypatch.setattr(
        hub_repo, "hf_hub_url", lambda **_kwargs: "https://example.test/file"
    )
    monkeypatch.setattr(hub_repo, "get_hf_file_metadata", lambda _url: Metadata())
    source = hub_repo.HuggingfaceTinystoriesRepository("noanabeshima/TinyStoriesV2")
    with pytest.raises(PublishedSizeMissingError, match="no size"):
        source.byte_count("train.jsonl")


def test_fetch_downloads_into_the_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch asks the Hub for the dataset file and writes it under the destination."""

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        local_dir: Path,
    ) -> str:
        assert repo_id == "noanabeshima/TinyStoriesV2"
        assert repo_type == "dataset"
        target = local_dir / filename
        target.write_bytes(b"abc")
        return str(target)

    monkeypatch.setattr(hub_repo, "hf_hub_download", fake_hf_hub_download)
    source = hub_repo.HuggingfaceTinystoriesRepository("noanabeshima/TinyStoriesV2")
    source.fetch("train.jsonl", tmp_path)
    assert (tmp_path / "train.jsonl").read_bytes() == b"abc"
