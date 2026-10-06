"""Use-case tests for placing TinyStoriesV2 files in raw storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tiny_stories_experiment.application.ports.published_dataset_source import (
    PublishedDatasetSource,
)
from tiny_stories_experiment.application.use_cases.download_dataset import (
    download_dataset,
)
from tiny_stories_experiment.composition import DEFAULT_RAW_DESTINATION, TINY_STORIES_V2
from tiny_stories_experiment.domain.dataset.dataset_spec import DatasetSpec
from tiny_stories_experiment.domain.errors import (
    DownloadSizeMismatchError,
    RawFileConflictError,
)
from tiny_stories_experiment.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)

if TYPE_CHECKING:
    from pathlib import Path


class RecordingSource(PublishedDatasetSource):
    """Published-file source that records fetch calls and writes a fixed payload."""

    def __init__(self, sizes: dict[str, int], payload: bytes) -> None:
        """Store the published sizes and the bytes a fetch writes.

        Args:
            sizes: Published byte count by file name.
            payload: Bytes written when a file is fetched.
        """
        self._sizes = sizes
        self._payload = payload
        self.fetched: list[str] = []

    def byte_count(self, filename: str) -> int:
        """Return the scripted published size.

        Args:
            filename: File name whose size was scripted.

        Returns:
            The scripted byte count.
        """
        return self._sizes[filename]

    def fetch(self, filename: str, destination: Path) -> None:
        """Record the fetch and write the scripted payload.

        Args:
            filename: File name to write.
            destination: Directory that receives the file.
        """
        self.fetched.append(filename)
        (destination / filename).write_bytes(self._payload)


def test_tiny_stories_v2_lands_in_raw_tiny_stories() -> None:
    """The wired dataset is TinyStoriesV2 and the raw directory is the planned path."""
    assert TINY_STORIES_V2.repo_id == "noanabeshima/TinyStoriesV2"
    assert TINY_STORIES_V2.source_url == (
        "https://huggingface.co/datasets/noanabeshima/TinyStoriesV2"
    )
    assert TINY_STORIES_V2.filenames == (
        "TinyStoriesV2-GPT4-train.jsonl",
        "TinyStoriesV2-GPT4-valid.jsonl",
    )
    assert DEFAULT_RAW_DESTINATION.parts[-3:] == ("data", "raw", "tiny_stories_raw")


def test_missing_file_is_downloaded(tmp_path: Path) -> None:
    """A missing JSONL file is fetched into the destination directory."""
    spec = DatasetSpec("repo", "https://example.test/data", ("train.jsonl",))
    source = RecordingSource({"train.jsonl": 3}, b"abc")
    report = download_dataset(spec, tmp_path, source, FilesystemDatasetRepository())
    assert report.downloaded == ("train.jsonl",)
    assert report.kept == ()
    assert source.fetched == ["train.jsonl"]
    assert (tmp_path / "train.jsonl").read_bytes() == b"abc"


def test_matching_local_file_is_kept(tmp_path: Path) -> None:
    """A local file with the published byte count is not downloaded again."""
    (tmp_path / "train.jsonl").write_bytes(b"abc")
    spec = DatasetSpec("repo", "https://example.test/data", ("train.jsonl",))
    source = RecordingSource({"train.jsonl": 3}, b"xyz")
    report = download_dataset(spec, tmp_path, source, FilesystemDatasetRepository())
    assert report.kept == ("train.jsonl",)
    assert report.downloaded == ()
    assert source.fetched == []
    assert (tmp_path / "train.jsonl").read_bytes() == b"abc"


def test_different_local_size_is_refused(tmp_path: Path) -> None:
    """A local file with a different size is left in place and not replaced."""
    local_file = tmp_path / "train.jsonl"
    local_file.write_bytes(b"partial")
    spec = DatasetSpec("repo", "https://example.test/data", ("train.jsonl",))
    source = RecordingSource({"train.jsonl": 3}, b"abc")
    with pytest.raises(RawFileConflictError, match="Remove the local file"):
        download_dataset(spec, tmp_path, source, FilesystemDatasetRepository())
    assert source.fetched == []
    assert local_file.read_bytes() == b"partial"


def test_downloaded_size_mismatch_raises(tmp_path: Path) -> None:
    """A mismatched download is removed so the next run can fetch the file."""
    spec = DatasetSpec("repo", "https://example.test/data", ("train.jsonl",))
    source = RecordingSource({"train.jsonl": 3}, b"ab")
    store = FilesystemDatasetRepository()
    with pytest.raises(DownloadSizeMismatchError, match="Downloaded"):
        download_dataset(spec, tmp_path, source, store)

    assert not (tmp_path / "train.jsonl").exists()
    retry = RecordingSource({"train.jsonl": 3}, b"abc")
    report = download_dataset(spec, tmp_path, retry, store)
    assert report.downloaded == ("train.jsonl",)
    assert report.kept == ()
    assert (tmp_path / "train.jsonl").read_bytes() == b"abc"
