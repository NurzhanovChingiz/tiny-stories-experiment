"""Behavior of configured dataset acquisition."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tiny_story.composition import acquire_dataset
from tiny_story.domain.errors import DownloadError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_default_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the default artifacts/download/logs tree inside the test directory.

    Args:
        tmp_path: Per-test directory.
        monkeypatch: Pytest patch helper used to change the working directory.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)


def _write_config(
    path: Path,
    *,
    destination: Path,
    cache_dir: Path,
    overwrite: bool = False,
) -> None:
    """Write a Hugging Face dataset configuration.

    Args:
        path: Configuration path.
        destination: Raw directory recorded in the file.
        cache_dir: Hugging Face cache recorded in the file.
        overwrite: Whether complete raw files are acquired again.

    Returns:
        None.
    """
    flag = "true" if overwrite else "false"
    path.write_text(
        f"""
dataset:
  name: other_corpus
  files:
    - notes.txt
  destination: {destination}
  overwrite: {flag}
  source:
    repository: some-org/other-corpus
    revision: abc123
    cache_dir: {cache_dir}
""",
        encoding="utf-8",
    )


def _patch_published_digest(monkeypatch: pytest.MonkeyPatch, digest: object) -> None:
    """Return a Hub etag without calling the live Hub.

    Args:
        monkeypatch: Pytest patch helper.
        digest: SHA-256 hex string, or a zero-argument callable that returns one.

    Returns:
        None.
    """

    def fake_metadata(*_args: object, **_kwargs: object) -> SimpleNamespace:
        etag = digest() if callable(digest) else digest
        return SimpleNamespace(etag=etag)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.get_hf_file_metadata",
        fake_metadata,
    )


def _sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of ``payload``.

    Args:
        payload: Bytes whose digest the fake Hub should publish.

    Returns:
        The lowercase SHA-256 hex digest.
    """
    return hashlib.sha256(payload).hexdigest()


def test_acquire_downloads_a_huggingface_dataset_from_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hugging Face settings come from configuration, including another dataset."""
    cached = tmp_path / "cached.txt"
    cached.write_bytes(b"corpus\n")
    seen: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        seen.update(kwargs)
        return str(cached)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    _patch_published_digest(monkeypatch, _sha256(b"corpus\n"))
    destination = tmp_path / "raw"
    cache = tmp_path / "cache"
    config = tmp_path / "dataset.yaml"
    _write_config(
        config,
        destination=tmp_path / "ignored",
        cache_dir=tmp_path / "ignored-cache",
    )

    outcome = acquire_dataset(config, destination=destination, cache_dir=cache)

    stored = destination / "notes.txt"
    assert stored.read_bytes() == b"corpus\n"
    assert outcome.written_paths == (stored,)
    assert seen["repo_id"] == "some-org/other-corpus"
    assert seen["filename"] == "notes.txt"
    assert seen["revision"] == "abc123"
    assert seen["cache_dir"] == cache
    assert seen["repo_type"] == "dataset"
    log_file = tmp_path / "artifacts" / "download" / "logs" / "other_corpus.log"
    log_text = log_file.read_text(encoding="utf-8")
    assert "Loaded dataset configuration" in log_text
    assert not (destination.parent / "logs").exists()


def test_acquire_keeps_a_complete_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete raw file is left in place and is not downloaded again."""
    cached = tmp_path / "cached.txt"
    cached.write_bytes(b"one\n")
    calls: list[str] = []

    def fake_download(**kwargs: object) -> str:
        filename = kwargs["filename"]
        assert isinstance(filename, str)
        calls.append(filename)
        return str(cached)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    _patch_published_digest(monkeypatch, _sha256(b"one\n"))
    destination = tmp_path / "raw"
    cache = tmp_path / "cache"
    config = tmp_path / "dataset.yaml"
    _write_config(config, destination=destination, cache_dir=cache)

    first = acquire_dataset(config)
    cached.write_bytes(b"two\n")
    second = acquire_dataset(config)

    stored = destination / "notes.txt"
    assert stored.read_bytes() == b"one\n"
    assert first.written_paths == (stored,)
    assert first.already_present_paths == ()
    assert second.written_paths == ()
    assert second.already_present_paths == (stored,)
    assert calls == ["notes.txt"]


def test_acquire_replaces_a_file_when_overwrite_is_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overwrite downloads the Hugging Face file again."""
    cached = tmp_path / "cached.txt"
    cached.write_bytes(b"one\n")

    def fake_download(**_kwargs: object) -> str:
        return str(cached)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    _patch_published_digest(monkeypatch, lambda: _sha256(cached.read_bytes()))
    destination = tmp_path / "raw"
    config = tmp_path / "dataset.yaml"
    _write_config(
        config,
        destination=destination,
        cache_dir=tmp_path / "cache",
        overwrite=True,
    )

    acquire_dataset(config)
    cached.write_bytes(b"two\n")
    second = acquire_dataset(config)

    stored = destination / "notes.txt"
    assert stored.read_bytes() == b"two\n"
    assert second.written_paths == (stored,)
    assert second.already_present_paths == ()


def test_acquire_does_not_publish_when_download_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Hugging Face download does not leave a raw or partial file."""

    def fake_download(**_kwargs: object) -> str:
        message = "offline"
        raise OSError(message)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    _patch_published_digest(monkeypatch, _sha256(b"one\n"))
    destination = tmp_path / "raw"
    config = tmp_path / "dataset.yaml"
    _write_config(config, destination=destination, cache_dir=tmp_path / "cache")

    with pytest.raises(DownloadError, match=r"Could not download notes\.txt"):
        acquire_dataset(config)

    assert not (destination / "notes.txt").exists()
    assert not (destination / "notes.txt.partial").exists()
    log_text = (
        tmp_path / "artifacts" / "download" / "logs" / "other_corpus.log"
    ).read_text(encoding="utf-8")
    assert "ERROR" in log_text
    assert "Acquisition failed for notes.txt" in log_text


def test_acquire_writes_the_run_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acquisition appends kept and stored files to the filesystem run log."""
    cached = tmp_path / "cached.txt"
    cached.write_bytes(b"corpus\n")

    def fake_download(**_kwargs: object) -> str:
        return str(cached)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )
    _patch_published_digest(monkeypatch, _sha256(b"already\n"))
    destination = tmp_path / "raw"
    destination.mkdir()
    log_path = tmp_path / "custom" / "run.log"
    config = tmp_path / "dataset.yaml"
    _write_config(config, destination=destination, cache_dir=tmp_path / "cache")
    (destination / "notes.txt").write_bytes(b"already\n")

    outcome = acquire_dataset(config, log_path=log_path)

    log_text = log_path.read_text(encoding="utf-8")
    assert outcome.already_present_paths == (destination / "notes.txt",)
    assert "Loaded dataset configuration" in log_text
    assert "Keeping existing raw file" in log_text
    assert "wrote 0 file(s), kept 1 file(s)" in log_text
    assert "INFO" in log_text
    assert not (tmp_path / "artifacts" / "download" / "logs").exists()
