"""Use-case tests for writing derived story text from raw JSONL files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tiny_stories_experiment.application.ports.raw_dataset_store import RawDatasetStore
from tiny_stories_experiment.application.use_cases.prepare_dataset import (
    prepare_dataset,
)
from tiny_stories_experiment.domain.dataset.dataset_spec import DatasetSpec
from tiny_stories_experiment.domain.dataset.text_sample import DatasetSplit
from tiny_stories_experiment.domain.errors import (
    RawDatasetChangedError,
    RawDatasetMissingError,
    RawStoryRecordError,
    UnknownDatasetSplitError,
)
from tiny_stories_experiment.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)
from tiny_stories_experiment.infrastructure.datasets.filesystem_processed_text_store import (
    FilesystemProcessedTextStore,
)
from tiny_stories_experiment.infrastructure.datasets.jsonl_story_source import (
    JsonlStorySource,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tiny_stories_experiment.application.results.prepare_outcome import (
        PrepareOutcome,
    )

TRAIN = "TinyStoriesV2-GPT4-train.jsonl"
VALID = "TinyStoriesV2-GPT4-valid.jsonl"
SPEC = DatasetSpec(
    "repo",
    "https://example.test/data",
    (TRAIN, VALID),
)


def _write_raw(directory: Path, filename: str, stories: list[str]) -> None:
    """Write story records as JSONL.

    Args:
        directory: Directory that receives the file.
        filename: JSONL file name.
        stories: Story text values, in order.
    """
    lines = [json.dumps({"text": story}) for story in stories]
    (directory / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bytes_by_name(directory: Path, names: tuple[str, ...]) -> dict[str, bytes]:
    """Read each named file in ``directory``.

    Args:
        directory: Directory that contains the files.
        names: File names to read.

    Returns:
        File contents keyed by name.
    """
    return {name: (directory / name).read_bytes() for name in names}


def _split_summary(
    outcome: PrepareOutcome,
) -> tuple[tuple[str, DatasetSplit, int, str], ...]:
    """Project each prepared split into comparable facts.

    Args:
        outcome: Report returned by preparation.

    Returns:
        ``(raw_filename, split, story_count, output_name)`` for each split, in order.
    """
    return tuple(
        (item.raw_filename, item.split, item.story_count, item.output_name)
        for item in outcome.splits
    )


def _jsonl_records(path: Path) -> list[object]:
    """Parse each JSON object in a derived JSONL file.

    Args:
        path: Derived JSONL file.

    Returns:
        One decoded JSON value per line, in file order.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _prepare(raw: Path, processed: Path, spec: DatasetSpec = SPEC) -> PrepareOutcome:
    """Prepare ``spec`` from ``raw`` into ``processed`` with the filesystem adapters.

    Args:
        raw: Directory of raw JSONL files.
        processed: Directory for derived JSONL files.
        spec: Dataset file names to prepare.

    Returns:
        The preparation report.
    """
    return prepare_dataset(
        spec,
        raw,
        processed,
        source=JsonlStorySource(),
        processed_store=FilesystemProcessedTextStore(),
        raw_store=FilesystemDatasetRepository(),
    )


def test_prepare_writes_split_files_and_leaves_raw_bytes(tmp_path: Path) -> None:
    """Derived JSONL keeps each story, including breaks, and raw bytes stay put."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    train_story = "Once upon a time.\nThe end."
    _write_raw(raw, TRAIN, [train_story, ""])
    _write_raw(raw, VALID, ["A second story."])
    raw_bytes = _bytes_by_name(raw, (TRAIN, VALID))

    outcome = _prepare(raw, processed)

    assert _bytes_by_name(raw, (TRAIN, VALID)) == raw_bytes
    assert _split_summary(outcome) == (
        (TRAIN, DatasetSplit.TRAIN, 2, "train.jsonl"),
        (VALID, DatasetSplit.VALID, 1, "valid.jsonl"),
    )
    assert _jsonl_records(processed / "train.jsonl") == [
        {"text": train_story},
        {"text": ""},
    ]


def test_second_prepare_replaces_derived_text(tmp_path: Path) -> None:
    """Running prepare again replaces the derived file with the current stories."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    _write_raw(raw, TRAIN, ["alpha"])
    _write_raw(raw, VALID, ["valid"])
    _prepare(raw, processed)

    _write_raw(raw, TRAIN, ["beta"])
    _prepare(raw, processed)

    assert _jsonl_records(processed / "train.jsonl") == [{"text": "beta"}]


def test_missing_raw_file_writes_nothing(tmp_path: Path) -> None:
    """An absent raw file stops preparation before the processed directory exists."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    _write_raw(raw, TRAIN, ["alpha"])

    with pytest.raises(RawDatasetMissingError, match=VALID):
        _prepare(raw, processed)

    assert not processed.exists()


def test_unknown_split_in_filename_writes_nothing(tmp_path: Path) -> None:
    """A raw file that does not name train or valid is not turned into derived text."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    _write_raw(raw, "notes.jsonl", ["alpha"])
    spec = DatasetSpec("repo", "https://example.test/data", ("notes.jsonl",))

    with pytest.raises(UnknownDatasetSplitError, match=r"notes\.jsonl"):
        _prepare(raw, processed, spec)

    assert list(processed.glob("*")) == []


def test_malformed_record_leaves_raw_bytes_and_no_derived_file(tmp_path: Path) -> None:
    """A non-story line is refused, the raw file is unchanged, and no output remains."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    source = raw / TRAIN
    source.write_text('{"text": 1}\n', encoding="utf-8")
    raw_bytes = source.read_bytes()
    spec = DatasetSpec("repo", "https://example.test/data", (TRAIN,))

    with pytest.raises(RawStoryRecordError, match="not a story record"):
        _prepare(raw, processed, spec)

    assert source.read_bytes() == raw_bytes
    assert not (processed / "train.jsonl").exists()
    assert not (processed / "train.jsonl.partial").exists()


class _ChangingSizeStore(RawDatasetStore):
    """Reports a different size after the first lookup of a file."""

    def __init__(self, sizes: dict[str, int]) -> None:
        """Store the sizes reported on the first lookup of each file.

        Args:
            sizes: Byte count by raw file name.
        """
        self._sizes = dict(sizes)
        self._seen: set[str] = set()

    def prepare(self, destination: Path) -> None:
        """Do not create a directory; this double only reports sizes.

        Args:
            destination: Unused raw directory from the port signature.
        """
        _ = destination

    def remove(self, destination: Path, filename: str) -> None:
        """Do not delete a file; this double only reports sizes.

        Args:
            destination: Unused raw directory from the port signature.
            filename: Unused raw file name from the port signature.
        """
        _ = destination, filename

    def byte_count(self, destination: Path, filename: str) -> int | None:
        """Return the scripted size, then one byte more on a later lookup.

        Args:
            destination: Unused raw directory from the port signature.
            filename: Raw file name whose size was scripted.

        Returns:
            The scripted size on the first lookup, otherwise one byte more.
        """
        _ = destination
        size = self._sizes[filename]
        if filename in self._seen:
            return size + 1
        self._seen.add(filename)
        return size


def test_changed_raw_size_is_refused(tmp_path: Path) -> None:
    """Preparation fails when a raw file's size is no longer the recorded count."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    _write_raw(raw, TRAIN, ["alpha"])
    spec = DatasetSpec("repo", "https://example.test/data", (TRAIN,))
    processed.mkdir()
    (processed / "train.jsonl").write_text("OLD\n", encoding="utf-8")
    store = _ChangingSizeStore({TRAIN: (raw / TRAIN).stat().st_size})

    with pytest.raises(RawDatasetChangedError, match="changed from"):
        prepare_dataset(
            spec,
            raw,
            processed,
            source=JsonlStorySource(),
            processed_store=FilesystemProcessedTextStore(),
            raw_store=store,
        )

    assert (processed / "train.jsonl").read_text(encoding="utf-8") == "OLD\n"
    assert not (processed / "train.jsonl.partial").exists()
    assert not (processed / "train.jsonl.previous").exists()


def test_publish_failure_restores_derived_files(tmp_path: Path) -> None:
    """A failed publish puts previous derived files back, including a missing one."""
    store = FilesystemProcessedTextStore()
    store.prepare(tmp_path)
    (tmp_path / "train.jsonl").write_text("OLD\n", encoding="utf-8")
    store.stage(tmp_path, DatasetSplit.TRAIN, ["alpha"])
    fresh = tmp_path / "fresh"
    store.prepare(fresh)
    store.stage(fresh, DatasetSplit.TRAIN, ["alpha"])

    with pytest.raises(FileNotFoundError, match=r"valid\.jsonl\.partial"):
        store.publish(tmp_path, (DatasetSplit.TRAIN, DatasetSplit.VALID))
    with pytest.raises(FileNotFoundError, match=r"valid\.jsonl\.partial"):
        store.publish(fresh, (DatasetSplit.TRAIN, DatasetSplit.VALID))

    assert (tmp_path / "train.jsonl").read_text(encoding="utf-8") == "OLD\n"
    assert not (fresh / "train.jsonl").exists()
    assert list(tmp_path.glob("*.partial")) == []
    assert list(tmp_path.glob("*.previous")) == []
    assert list(fresh.glob("*.partial")) == []
    assert list(fresh.glob("*.previous")) == []


def test_later_split_failure_keeps_previous_derived_files(tmp_path: Path) -> None:
    """A bad later split leaves every previous derived file byte-for-byte."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    _write_raw(raw, TRAIN, ["alpha"])
    (raw / VALID).write_text("not-json\n", encoding="utf-8")
    (processed / "train.jsonl").write_text("OLD-TRAIN\n", encoding="utf-8")
    (processed / "valid.jsonl").write_text("OLD-VALID\n", encoding="utf-8")

    with pytest.raises(RawStoryRecordError, match="not a story record"):
        _prepare(raw, processed)

    assert (processed / "train.jsonl").read_text(encoding="utf-8") == "OLD-TRAIN\n"
    assert (processed / "valid.jsonl").read_text(encoding="utf-8") == "OLD-VALID\n"
    assert list(processed.glob("*.partial")) == []
    assert list(processed.glob("*.previous")) == []


def test_stage_error_stays_primary_when_raw_size_changes(tmp_path: Path) -> None:
    """A bad story line stays the raised error when the raw size also changes."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    source = raw / TRAIN
    source.write_text('{"text": 1}\n', encoding="utf-8")
    spec = DatasetSpec("repo", "https://example.test/data", (TRAIN,))
    store = _ChangingSizeStore({TRAIN: source.stat().st_size})

    with pytest.raises(RawStoryRecordError, match="not a story record") as caught:
        prepare_dataset(
            spec,
            raw,
            processed,
            source=JsonlStorySource(),
            processed_store=FilesystemProcessedTextStore(),
            raw_store=store,
        )

    assert isinstance(caught.value.__cause__, RawDatasetChangedError)
    assert not (processed / "train.jsonl").exists()
    assert not (processed / "train.jsonl.partial").exists()
