"""Behavior of the filesystem raw dataset store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tiny_story.domain.errors import InvalidRawFilenameError, RawStorageError
from tiny_story.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)

_PUBLICATION = "publication-a"
_OTHER_PUBLICATION = "publication-b"


def _sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 of one payload.

    Args:
        payload: Bytes to hash.

    Returns:
        The lowercase SHA-256 hex digest.
    """
    return hashlib.sha256(payload).hexdigest()


def test_store_copies_bytes_and_leaves_the_published_file_unchanged(
    tmp_path: Path,
) -> None:
    """The raw file matches the published bytes, and the source file stays put."""
    published = tmp_path / "published.jsonl"
    payload = b'{"text":"story"}\n'
    published.write_bytes(payload)
    store = FilesystemDatasetRepository(tmp_path / "raw")

    stored = store.store_unchanged(
        "story.jsonl", published, _sha256(payload), _PUBLICATION
    )
    published.write_bytes(b'{"text":"changed"}\n')

    assert stored.read_bytes() == payload
    assert store.existing_file("story.jsonl", _sha256(payload), _PUBLICATION) == stored
    assert published.read_bytes() == b'{"text":"changed"}\n'


def test_two_stores_keep_separate_files(tmp_path: Path) -> None:
    """Writing through one store does not create a file in the other."""
    published = tmp_path / "published.jsonl"
    payload = b"one\n"
    published.write_bytes(payload)
    first = FilesystemDatasetRepository(tmp_path / "first")
    second = FilesystemDatasetRepository(tmp_path / "second")

    stored = first.store_unchanged(
        "story.jsonl", published, _sha256(payload), _PUBLICATION
    )

    assert stored == tmp_path / "first" / "story.jsonl"
    assert second.existing_file("story.jsonl", _sha256(payload), _PUBLICATION) is None
    assert second.directory == tmp_path / "second"


def test_store_rejects_a_filename_outside_the_raw_directory(tmp_path: Path) -> None:
    """A path-shaped name does not write outside the raw directory."""
    published = tmp_path / "published.jsonl"
    payload = b"one\n"
    published.write_bytes(payload)
    store = FilesystemDatasetRepository(tmp_path / "raw")

    with pytest.raises(InvalidRawFilenameError, match="single file name"):
        store.store_unchanged(
            "../outside.jsonl",
            published,
            _sha256(payload),
            _PUBLICATION,
        )

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
        store.store_unchanged("story.jsonl", published_dir, "0" * 64, _PUBLICATION)

    assert not (raw / "story.jsonl").exists()
    assert not (raw / "story.jsonl.partial").exists()
    assert sorted(path.name for path in raw.iterdir()) == [".holds"]


def test_store_preserves_a_dataset_file_named_partial(tmp_path: Path) -> None:
    """Storing one file does not delete another file whose name ends in .partial."""
    published = tmp_path / "published.jsonl"
    payload = b"notes\n"
    published.write_bytes(payload)
    raw = tmp_path / "raw"
    raw.mkdir()
    legitimate = raw / "notes.txt.partial"
    legitimate.write_bytes(b"kept\n")

    stored = FilesystemDatasetRepository(raw).store_unchanged(
        "notes.txt",
        published,
        _sha256(payload),
        _PUBLICATION,
    )

    assert stored.read_bytes() == payload
    assert legitimate.read_bytes() == b"kept\n"
    assert sorted(path.name for path in raw.iterdir()) == [
        ".digests",
        ".holds",
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
        FilesystemDatasetRepository(raw).store_unchanged(
            "notes.txt",
            published_dir,
            "0" * 64,
            _PUBLICATION,
        )

    assert existing.read_bytes() == b"old\n"
    assert sorted(path.name for path in raw.iterdir()) == [".holds", "notes.txt"]


def test_store_replaces_an_existing_regular_file(tmp_path: Path) -> None:
    """A successful store replaces the previous bytes and leaves no temporary file."""
    published = tmp_path / "published.txt"
    payload = b"new\n"
    published.write_bytes(payload)
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "notes.txt"
    existing.write_bytes(b"old\n")

    stored = FilesystemDatasetRepository(raw).store_unchanged(
        "notes.txt",
        published,
        _sha256(payload),
        _PUBLICATION,
    )

    assert stored == existing
    assert existing.read_bytes() == payload
    assert sorted(path.name for path in raw.iterdir()) == [
        ".digests",
        ".holds",
        "notes.txt",
    ]


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
        FilesystemDatasetRepository(raw).existing_file(
            "story.jsonl",
            _sha256(b"secret\n"),
            _PUBLICATION,
        )

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
        FilesystemDatasetRepository(raw).store_unchanged(
            "story.jsonl",
            published,
            _sha256(b"incoming\n"),
            _PUBLICATION,
        )

    assert outside.read_bytes() == b"secret\n"
    assert link.is_symlink()
    assert sorted(path.name for path in raw.iterdir()) == [".holds", "story.jsonl"]


def test_existing_file_rejects_a_digest_that_does_not_match(tmp_path: Path) -> None:
    """A different digest is not a match, and a missing file leaves no record."""
    raw = tmp_path / "raw"
    raw.mkdir()
    empty = raw / "story.jsonl"
    empty.write_bytes(b"")
    store = FilesystemDatasetRepository(raw)

    assert store.existing_file("story.jsonl", _sha256(b"other\n"), _PUBLICATION) is None
    assert empty.read_bytes() == b""
    assert (
        store.existing_file("missing.jsonl", _sha256(b"other\n"), _PUBLICATION) is None
    )
    assert not (raw / ".digests" / "missing.jsonl").exists()


def test_digest_mismatch_clears_the_publication_record(tmp_path: Path) -> None:
    """A mismatched digest removes the record so a later verification does not keep it."""
    raw = tmp_path / "raw"
    payload = b"notes\n"
    published = tmp_path / "published.txt"
    published.write_bytes(payload)
    store = FilesystemDatasetRepository(raw)
    stored = store.store_unchanged(
        "notes.txt", published, _sha256(payload), _PUBLICATION
    )
    stored.write_bytes(b"changed\n")

    assert store.existing_file("notes.txt", _sha256(payload), _PUBLICATION) is None
    assert store.verified_file("notes.txt", _PUBLICATION) is None
    assert stored.read_bytes() == b"changed\n"
    assert not (raw / ".digests" / "notes.txt").exists()


def test_missing_file_leaves_its_publication_record(tmp_path: Path) -> None:
    """Looking up an absent file does not delete a record that is not there."""
    raw = tmp_path / "raw"
    digests = raw / ".digests"
    digests.mkdir(parents=True)
    record = digests / "notes.txt"
    record.write_text(
        json.dumps({"publication_id": _PUBLICATION, "sha256": _sha256(b"notes\n")}),
        encoding="utf-8",
    )
    store = FilesystemDatasetRepository(raw)

    assert store.existing_file("notes.txt", _sha256(b"notes\n"), _PUBLICATION) is None
    assert record.is_file()


def test_reading_notes_does_not_touch_a_digest_named_like_a_hold(
    tmp_path: Path,
) -> None:
    """A hold-shaped digest name stays beside the file that was actually read."""
    raw = tmp_path / "raw"
    raw.mkdir()
    notes = raw / "notes.txt"
    notes.write_bytes(b"notes\n")
    digests = raw / ".digests"
    digests.mkdir()
    decoy = digests / ".notes.txt.hold"
    decoy.write_bytes(b"decoy\n")
    sibling = raw / ".notes.txt.hold"
    sibling.write_bytes(b"sibling\n")

    found = FilesystemDatasetRepository(raw).existing_file(
        "notes.txt",
        _sha256(b"notes\n"),
        _PUBLICATION,
    )

    assert found == notes
    assert decoy.read_bytes() == b"decoy\n"
    assert sibling.read_bytes() == b"sibling\n"


def test_symlinked_parent_is_rejected_before_publication(tmp_path: Path) -> None:
    """A symlinked ancestor is rejected and its target stays empty."""
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    published = tmp_path / "published.txt"
    published.write_bytes(b"notes\n")
    store = FilesystemDatasetRepository(linked / "raw")

    with pytest.raises(RawStorageError, match="symbolic link"):
        store.store_unchanged("notes.txt", published, _sha256(b"notes\n"), _PUBLICATION)

    assert list(outside.iterdir()) == []


def test_unparsable_digest_record_is_not_treated_as_missing(tmp_path: Path) -> None:
    """An unreadable digest record raises and leaves a held copy in place."""
    raw = tmp_path / "raw"
    raw.mkdir()
    digests = raw / ".digests"
    digests.mkdir()
    record = digests / "notes.txt"
    record.write_text("not-json", encoding="utf-8")
    holds = raw / ".holds" / "files"
    holds.mkdir(parents=True)
    hold = holds / "notes.txt"
    hold.write_bytes(b"held\n")
    store = FilesystemDatasetRepository(raw)

    with pytest.raises(RawStorageError, match=r"publication record for notes\.txt"):
        store.verified_file("notes.txt", _PUBLICATION)

    assert hold.read_bytes() == b"held\n"
    assert record.read_text(encoding="utf-8") == "not-json"


def test_hash_oserror_is_a_storage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hash read error is reported as raw storage failure."""
    raw = tmp_path / "raw"
    raw.mkdir()
    notes = raw / "notes.txt"
    notes.write_bytes(b"notes\n")

    def fail_open(*_args: object, **_kwargs: object) -> object:
        raise OSError(13)

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(RawStorageError, match=r"notes\.txt") as caught:
        FilesystemDatasetRepository(raw).existing_file(
            "notes.txt",
            _sha256(b"notes\n"),
            _PUBLICATION,
        )

    assert isinstance(caught.value.__cause__, OSError)


def test_recovery_restores_a_hold_without_using_the_lock_path(tmp_path: Path) -> None:
    """A held copy is restored from the hold-files directory, not from the lock."""
    raw = tmp_path / "raw"
    holds = raw / ".holds" / "files"
    holds.mkdir(parents=True)
    hold = holds / "publish.lock"
    hold.write_bytes(b"held\n")
    digests = raw / ".digests"
    digests.mkdir()
    (digests / "publish.lock").write_text(
        json.dumps(
            {"publication_id": _PUBLICATION, "sha256": _sha256(b"held\n")},
        ),
        encoding="utf-8",
    )
    store = FilesystemDatasetRepository(raw)

    restored = store.verified_file("publish.lock", _PUBLICATION)

    assert restored == raw / "publish.lock"
    assert restored is not None
    assert restored.read_bytes() == b"held\n"
    assert not hold.exists()
    lock = raw / ".holds" / "lock"
    assert lock.is_file()
    assert lock.resolve() != restored.resolve()


def test_lock_path_is_not_a_hold_path(tmp_path: Path) -> None:
    """The publication lock is not the hold path of a legal raw filename."""
    published = tmp_path / "published.txt"
    published.write_bytes(b"lock\n")
    raw = tmp_path / "raw"
    store = FilesystemDatasetRepository(raw)

    stored = store.store_unchanged("lock", published, _sha256(b"lock\n"), _PUBLICATION)

    assert stored == raw / "lock"
    assert (raw / ".holds" / "files" / "lock").is_file() is False
    assert (raw / ".holds" / "lock").is_file()
    assert store.existing_file("lock", _sha256(b"other\n"), _OTHER_PUBLICATION) is None
