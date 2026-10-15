"""Behavior of acquiring a dataset into raw storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tiny_story.application.ports.dataset_source import DatasetSource
from tiny_story.application.ports.run_logger import RunLogger
from tiny_story.application.use_cases.download_dataset import (
    ExistingFilePolicy,
    download_dataset,
)
from tiny_story.domain.dataset.dataset_spec import DatasetSpec
from tiny_story.domain.errors import (
    DownloadError,
    InvalidDatasetSpecError,
    InvalidRawFilenameError,
    RawStorageError,
)
from tiny_story.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)

if TYPE_CHECKING:
    from pathlib import Path


SPEC = DatasetSpec(
    name="tiny_stories",
    filenames=(
        "TinyStoriesV2-GPT4-train.jsonl",
        "TinyStoriesV2-GPT4-valid.jsonl",
    ),
)


class _MapSource(DatasetSource):
    """Returns prepared files and records which names were fetched."""

    def __init__(self, files: dict[str, Path]) -> None:
        """Store the files this source can return.

        Args:
            files: File name to local source path.

        Returns:
            None.
        """
        self._files = files
        self.fetched: list[str] = []

    def fetch(self, filename: str) -> Path:
        """Return one prepared file.

        Args:
            filename: File name to fetch.

        Returns:
            The prepared path.
        """
        self.fetched.append(filename)
        return self._files[filename]


class _FailingAfter(DatasetSource):
    """Returns prepared files until the named file fails."""

    def __init__(self, files: dict[str, Path], fail_at: str) -> None:
        """Store the files that succeed and the name that fails.

        Args:
            files: File name to local source path.
            fail_at: File name that raises a download error.

        Returns:
            None.
        """
        self._files = files
        self._fail_at = fail_at
        self.fetched: list[str] = []

    def fetch(self, filename: str) -> Path:
        """Return one prepared file, or fail at the named file.

        Args:
            filename: File name to fetch.

        Returns:
            The prepared path.

        Raises:
            DownloadError: This is the file selected to fail.
        """
        self.fetched.append(filename)
        if filename == self._fail_at:
            message = f"Could not acquire {filename}"
            raise DownloadError(message)
        return self._files[filename]


class _RecordingLogger(RunLogger):
    """Stores log calls so a test can see what the run recorded."""

    def __init__(self) -> None:
        """Start with an empty record.

        Returns:
            None.
        """
        self.records: list[tuple[str, str]] = []

    def log(self, message: str, *, level: str = "info") -> None:
        """Store one recorded message.

        Args:
            message: Fact to record.
            level: Severity supplied by the caller.

        Returns:
            None.
        """
        self.records.append((level, message))

    def close(self) -> None:
        """Close is unused because this logger keeps records in memory.

        Returns:
            None.
        """


class _FailingSource(DatasetSource):
    """Fails every fetch."""

    def fetch(self, filename: str) -> Path:
        """Raise a download error.

        Args:
            filename: File name that would have been fetched.

        Returns:
            This method does not return.

        Raises:
            DownloadError: The fetch failed.
        """
        message = f"Could not acquire {filename}"
        raise DownloadError(message)


def test_download_copies_missing_files_and_leaves_existing_bytes(
    tmp_path: Path,
) -> None:
    """An existing raw file stays as it is, and a missing file is copied."""
    published_train = tmp_path / "published-train.jsonl"
    published_valid = tmp_path / "published-valid.jsonl"
    published_train.write_bytes(b'{"text":"train"}\n')
    published_valid.write_bytes(b'{"text":"valid"}\n')
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "TinyStoriesV2-GPT4-train.jsonl"
    existing.write_bytes(b'{"text":"kept"}\n')
    source = _MapSource(
        {
            "TinyStoriesV2-GPT4-train.jsonl": published_train,
            "TinyStoriesV2-GPT4-valid.jsonl": published_valid,
        },
    )

    outcome = download_dataset(SPEC, source, FilesystemDatasetRepository(raw))

    stored_valid = raw / "TinyStoriesV2-GPT4-valid.jsonl"
    assert source.fetched == ["TinyStoriesV2-GPT4-valid.jsonl"]
    assert existing.read_bytes() == b'{"text":"kept"}\n'
    assert stored_valid.read_bytes() == b'{"text":"valid"}\n'
    assert published_valid.read_bytes() == b'{"text":"valid"}\n'
    assert outcome.directory == raw
    assert outcome.written_paths == (stored_valid,)
    assert outcome.already_present_paths == (existing,)


def test_download_replaces_an_empty_raw_file(tmp_path: Path) -> None:
    """An empty raw file is not treated as a finished download."""
    published = tmp_path / "published.txt"
    published.write_bytes(b"notes\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    empty = raw / "notes.txt"
    empty.write_bytes(b"")
    spec = DatasetSpec(name="custom_corpus", filenames=("notes.txt",))
    source = _MapSource({"notes.txt": published})

    outcome = download_dataset(spec, source, FilesystemDatasetRepository(raw))

    assert source.fetched == ["notes.txt"]
    assert empty.read_bytes() == b"notes\n"
    assert outcome.written_paths == (empty,)
    assert outcome.already_present_paths == ()


def test_download_replaces_a_complete_file_when_asked(tmp_path: Path) -> None:
    """Replace acquires a file again and reports it as written."""
    published = tmp_path / "published.txt"
    published.write_bytes(b"new\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "notes.txt"
    existing.write_bytes(b"old\n")
    spec = DatasetSpec(name="custom_corpus", filenames=("notes.txt",))
    source = _MapSource({"notes.txt": published})

    outcome = download_dataset(
        spec,
        source,
        FilesystemDatasetRepository(raw),
        existing=ExistingFilePolicy.REPLACE,
    )

    assert source.fetched == ["notes.txt"]
    assert existing.read_bytes() == b"new\n"
    assert outcome.written_paths == (existing,)
    assert outcome.already_present_paths == ()


def test_download_keeps_both_a_file_and_its_partial_name(tmp_path: Path) -> None:
    """Acquiring notes.txt does not destroy a real notes.txt.partial file."""
    notes = tmp_path / "notes"
    partial = tmp_path / "partial"
    notes.write_bytes(b"notes\n")
    partial.write_bytes(b"partial\n")
    raw = tmp_path / "raw"
    spec = DatasetSpec(
        name="custom_corpus",
        filenames=("notes.txt.partial", "notes.txt"),
    )
    source = _MapSource(
        {"notes.txt.partial": partial, "notes.txt": notes},
    )

    outcome = download_dataset(spec, source, FilesystemDatasetRepository(raw))

    stored_partial = raw / "notes.txt.partial"
    stored_notes = raw / "notes.txt"
    assert stored_partial.read_bytes() == b"partial\n"
    assert stored_notes.read_bytes() == b"notes\n"
    assert outcome.written_paths == (stored_partial, stored_notes)
    assert sorted(path.name for path in raw.iterdir()) == [
        "notes.txt",
        "notes.txt.partial",
    ]


def test_published_file_remains_when_a_later_file_fails(tmp_path: Path) -> None:
    """A later fetch failure leaves the earlier published file in place."""
    published = tmp_path / "published-train.jsonl"
    published.write_bytes(b"train\n")
    raw = tmp_path / "raw"
    spec = DatasetSpec(name="custom_corpus", filenames=("train.jsonl", "valid.jsonl"))
    source = _FailingAfter({"train.jsonl": published}, fail_at="valid.jsonl")

    with pytest.raises(DownloadError, match=r"Could not acquire valid\.jsonl"):
        download_dataset(spec, source, FilesystemDatasetRepository(raw))

    stored = raw / "train.jsonl"
    assert stored.read_bytes() == b"train\n"
    assert not (raw / "valid.jsonl").exists()
    assert list(raw.iterdir()) == [stored]
    assert source.fetched == ["train.jsonl", "valid.jsonl"]


def test_replace_keeps_the_original_when_the_new_copy_fails(tmp_path: Path) -> None:
    """A failed overwrite does not replace the existing raw bytes."""
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "notes.txt"
    existing.write_bytes(b"old\n")
    source_dir = tmp_path / "not-a-file"
    source_dir.mkdir()
    spec = DatasetSpec(name="custom_corpus", filenames=("notes.txt",))

    with pytest.raises(RawStorageError, match=r"Could not store raw file notes\.txt"):
        download_dataset(
            spec,
            _MapSource({"notes.txt": source_dir}),
            FilesystemDatasetRepository(raw),
            existing=ExistingFilePolicy.REPLACE,
        )

    assert existing.read_bytes() == b"old\n"
    assert list(raw.iterdir()) == [existing]


def test_download_does_not_store_a_file_when_acquisition_fails(tmp_path: Path) -> None:
    """A source failure does not publish a raw file or a success outcome."""
    raw = tmp_path / "raw"
    spec = DatasetSpec(name="custom_corpus", filenames=("notes.txt",))

    with pytest.raises(DownloadError, match=r"Could not acquire notes\.txt"):
        download_dataset(spec, _FailingSource(), FilesystemDatasetRepository(raw))

    assert not (raw / "notes.txt").exists()


def test_download_rejects_an_unsafe_filename_before_fetch(tmp_path: Path) -> None:
    """A path-shaped filename is rejected and nothing is fetched."""
    spec = DatasetSpec(name="corpus", filenames=("../escape.jsonl",))
    source = _MapSource({})

    with pytest.raises(InvalidRawFilenameError, match="single file name"):
        download_dataset(spec, source, FilesystemDatasetRepository(tmp_path / "raw"))

    assert source.fetched == []


def test_download_records_kept_and_stored_files(tmp_path: Path) -> None:
    """A run log names the kept file, the fetch, the stored file, and the summary."""
    published_valid = tmp_path / "published-valid.jsonl"
    published_valid.write_bytes(b'{"text":"valid"}\n')
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "TinyStoriesV2-GPT4-train.jsonl"
    existing.write_bytes(b'{"text":"kept"}\n')
    source = _MapSource(
        {
            "TinyStoriesV2-GPT4-train.jsonl": tmp_path / "unused-train.jsonl",
            "TinyStoriesV2-GPT4-valid.jsonl": published_valid,
        },
    )
    run_logger = _RecordingLogger()

    download_dataset(
        SPEC,
        source,
        FilesystemDatasetRepository(raw),
        run_logger=run_logger,
    )

    messages = [message for _level, message in run_logger.records]
    assert any("using keep" in message for message in messages)
    assert f"Keeping existing raw file {existing}" in messages
    assert "Fetching TinyStoriesV2-GPT4-valid.jsonl" in messages
    assert f"Stored raw file {raw / 'TinyStoriesV2-GPT4-valid.jsonl'}" in messages
    assert any("wrote 1 file(s), kept 1 file(s)" in message for message in messages)
    assert source.fetched == ["TinyStoriesV2-GPT4-valid.jsonl"]


def test_download_records_a_failed_acquisition(tmp_path: Path) -> None:
    """A failed fetch is recorded at error level and still raises."""
    raw = tmp_path / "raw"
    spec = DatasetSpec(name="custom_corpus", filenames=("notes.txt",))
    run_logger = _RecordingLogger()

    with pytest.raises(DownloadError, match=r"Could not acquire notes\.txt"):
        download_dataset(
            spec,
            _FailingSource(),
            FilesystemDatasetRepository(raw),
            run_logger=run_logger,
        )

    assert (
        "error",
        "Acquisition failed for notes.txt: Could not acquire notes.txt",
    ) in run_logger.records


def test_download_rejects_a_blank_name_before_fetch(tmp_path: Path) -> None:
    """A blank dataset name is rejected and nothing is fetched."""
    spec = DatasetSpec(name=" ", filenames=("train.jsonl",))
    source = _MapSource({})

    with pytest.raises(InvalidDatasetSpecError, match="Dataset name"):
        download_dataset(spec, source, FilesystemDatasetRepository(tmp_path / "raw"))

    assert source.fetched == []
