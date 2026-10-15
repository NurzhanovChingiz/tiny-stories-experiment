"""Behavior of the filesystem raw dataset store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tiny_story.domain.errors import InvalidRawFilenameError, RawStorageError
from tiny_story.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_store_copies_bytes_and_leaves_the_published_file_unchanged(
    tmp_path: Path,
) -> None:
    """The raw file matches the published bytes, and the source file stays put."""
    published = tmp_path / "published.jsonl"
    published.write_bytes(b'{"text":"story"}\n')
    store = FilesystemDatasetRepository(tmp_path / "raw")

    stored = store.store_unchanged("story.jsonl", published)
    published.write_bytes(b'{"text":"changed"}\n')

    assert stored.read_bytes() == b'{"text":"story"}\n'
    assert store.existing_file("story.jsonl") == stored
    assert published.read_bytes() == b'{"text":"changed"}\n'


def test_two_stores_keep_separate_files(tmp_path: Path) -> None:
    """Writing through one store does not create a file in the other."""
    published = tmp_path / "published.jsonl"
    published.write_bytes(b"one\n")
    first = FilesystemDatasetRepository(tmp_path / "first")
    second = FilesystemDatasetRepository(tmp_path / "second")

    stored = first.store_unchanged("story.jsonl", published)

    assert stored == tmp_path / "first" / "story.jsonl"
    assert second.existing_file("story.jsonl") is None
    assert second.directory == tmp_path / "second"


def test_store_rejects_a_filename_outside_the_raw_directory(tmp_path: Path) -> None:
    """A path-shaped name does not write outside the raw directory."""
    published = tmp_path / "published.jsonl"
    published.write_bytes(b"one\n")
    store = FilesystemDatasetRepository(tmp_path / "raw")

    with pytest.raises(InvalidRawFilenameError, match="single file name"):
        store.store_unchanged("../outside.jsonl", published)

    assert not (tmp_path / "outside.jsonl").exists()


def test_store_removes_a_partial_file_when_the_copy_fails(tmp_path: Path) -> None:
    """A failed copy does not leave a temporary or final raw file."""
    published_dir = tmp_path / "not-a-file"
    published_dir.mkdir()
    raw = tmp_path / "raw"
    store = FilesystemDatasetRepository(raw)

    with pytest.raises(
        RawStorageError,
        match="Could not store raw file story\\.jsonl",
    ):
        store.store_unchanged("story.jsonl", published_dir)

    assert not (raw / "story.jsonl").exists()
    assert not (raw / "story.jsonl.partial").exists()
    assert list(raw.iterdir()) == []


def test_store_preserves_a_dataset_file_named_partial(tmp_path: Path) -> None:
    """Storing one file does not delete another file whose name ends in .partial."""
    published = tmp_path / "published.jsonl"
    published.write_bytes(b"notes\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    legitimate = raw / "notes.txt.partial"
    legitimate.write_bytes(b"kept\n")

    stored = FilesystemDatasetRepository(raw).store_unchanged("notes.txt", published)

    assert stored.read_bytes() == b"notes\n"
    assert legitimate.read_bytes() == b"kept\n"
    assert sorted(path.name for path in raw.iterdir()) == [
        "notes.txt",
        "notes.txt.partial",
    ]


def test_failed_copy_leaves_the_existing_file_unchanged(tmp_path: Path) -> None:
    """A failed replacement does not publish an incomplete file."""
    published_dir = tmp_path / "not-a-file"
    published_dir.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "notes.txt"
    existing.write_bytes(b"old\n")

    with pytest.raises(RawStorageError, match=r"Could not store raw file notes\.txt"):
        FilesystemDatasetRepository(raw).store_unchanged("notes.txt", published_dir)

    assert existing.read_bytes() == b"old\n"
    assert list(raw.iterdir()) == [existing]


def test_store_replaces_an_existing_regular_file(tmp_path: Path) -> None:
    """A successful store replaces the previous bytes and leaves no temporary file."""
    published = tmp_path / "published.txt"
    published.write_bytes(b"new\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "notes.txt"
    existing.write_bytes(b"old\n")

    stored = FilesystemDatasetRepository(raw).store_unchanged("notes.txt", published)

    assert stored == existing
    assert existing.read_bytes() == b"new\n"
    assert list(raw.iterdir()) == [existing]


def test_existing_file_rejects_a_symlink_outside_the_raw_directory(
    tmp_path: Path,
) -> None:
    """A symbolic link is not treated as a finished raw file."""
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"secret\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    link = raw / "story.jsonl"
    link.symlink_to(outside)

    with pytest.raises(RawStorageError, match="symbolic link"):
        FilesystemDatasetRepository(raw).existing_file("story.jsonl")

    assert outside.read_bytes() == b"secret\n"
    assert link.is_symlink()


def test_store_rejects_a_symlink_and_leaves_the_external_file(tmp_path: Path) -> None:
    """Publishing refuses a destination symlink and does not change its target."""
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"secret\n")
    published = tmp_path / "published.jsonl"
    published.write_bytes(b"incoming\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    link = raw / "story.jsonl"
    link.symlink_to(outside)

    with pytest.raises(RawStorageError, match="symbolic link"):
        FilesystemDatasetRepository(raw).store_unchanged("story.jsonl", published)

    assert outside.read_bytes() == b"secret\n"
    assert link.is_symlink()
    assert list(raw.iterdir()) == [link]


def test_existing_file_ignores_an_empty_file(tmp_path: Path) -> None:
    """An empty file is not a completed raw download."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "story.jsonl").write_bytes(b"")

    assert FilesystemDatasetRepository(raw).existing_file("story.jsonl") is None
