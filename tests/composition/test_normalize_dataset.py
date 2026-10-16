"""Behavior of configured story cleaning and normalization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tiny_story.composition import clean_dataset, normalize_dataset
from tiny_story.domain.errors import CleanTextError, NormalizeTextError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_default_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep default artifacts/clean and artifacts/normalize logs in the test directory.

    Args:
        tmp_path: Per-test directory.
        monkeypatch: Pytest patch helper used to change the working directory.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)


_CLEAN = """\
enabled: true
save_processed: true
page_header_pattern: '^Page'
junk_line_pattern: '^[/\\\\-]+$'
reflow_hard_wrapped_lines: true
join_hyphenated_line_breaks: true
"""

_NORMALIZE = """\
enabled: true
save_processed: true
sentence_delimiter: '|'
"""


def _dataset_config(path: Path, raw: Path, cache: Path) -> None:
    """Write a dataset configuration that names one local file.

    Args:
        path: Configuration path.
        raw: Raw directory recorded in the file.
        cache: Hugging Face cache recorded in the file.

    Returns:
        None.
    """
    path.write_text(
        f"""
dataset:
  name: other_corpus
  files:
    - valid.jsonl
  destination: {raw}
  overwrite: false
  source:
    repository: some-org/other-corpus
    revision: abc123
    cache_dir: {cache}
""",
        encoding="utf-8",
    )


def test_clean_then_normalize_writes_processed_text_outside_raw(tmp_path: Path) -> None:
    """Cleaning writes interim text, and normalization writes the delimiter."""
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "valid.jsonl"
    original = '{"text": "hello\\nworld."}\n'
    source.write_text(original, encoding="utf-8")
    dataset_config = tmp_path / "dataset.yaml"
    _dataset_config(dataset_config, raw, tmp_path / "cache")
    clean_config = tmp_path / "clean_text.yaml"
    clean_config.write_text(_CLEAN, encoding="utf-8")
    normalize_config = tmp_path / "normalize_text.yaml"
    normalize_config.write_text(_NORMALIZE, encoding="utf-8")
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"

    cleaned = clean_dataset(dataset_config, clean_config, output_dir=interim)
    written = normalize_dataset(
        dataset_config,
        normalize_config,
        input_dir=interim,
        output_dir=processed,
    )

    assert cleaned == (interim / "valid.jsonl",)
    assert written == (processed / "valid.jsonl",)
    assert source.read_text(encoding="utf-8") == original
    interim_text = json.loads((interim / "valid.jsonl").read_text(encoding="utf-8"))[
        "text"
    ]
    processed_text = json.loads(
        (processed / "valid.jsonl").read_text(encoding="utf-8")
    )["text"]
    assert "\n" not in interim_text
    assert ".|" not in interim_text
    assert processed_text.endswith(".|")
    _assert_default_run_logs(tmp_path)


def _assert_default_run_logs(tmp_path: Path) -> None:
    """Require the default clean and normalize logs to record the written files.

    Args:
        tmp_path: Per-test directory that holds ``artifacts``.

    Returns:
        None.
    """
    clean_log = (
        tmp_path / "artifacts" / "clean" / "logs" / "other_corpus.log"
    ).read_text(encoding="utf-8")
    normalize_log = (
        tmp_path / "artifacts" / "normalize" / "logs" / "other_corpus.log"
    ).read_text(encoding="utf-8")
    assert "Loaded cleaning configuration" in clean_log
    assert "Cleaning valid.jsonl" in clean_log
    assert "Wrote cleaned file" in clean_log
    assert "Cleaned other_corpus: wrote 1 file(s)" in clean_log
    assert "Loaded normalization configuration" in normalize_log
    assert "Normalizing valid.jsonl" in normalize_log
    assert "Wrote normalized file" in normalize_log
    assert "Normalized other_corpus: wrote 1 file(s)" in normalize_log


def test_clean_and_normalize_use_an_explicit_log_path(tmp_path: Path) -> None:
    """An explicit log path receives the run record and the default is unused."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "valid.jsonl").write_text('{"text": "hello."}\n', encoding="utf-8")
    dataset_config = tmp_path / "dataset.yaml"
    _dataset_config(dataset_config, raw, tmp_path / "cache")
    clean_config = tmp_path / "clean_text.yaml"
    clean_config.write_text(_CLEAN, encoding="utf-8")
    normalize_config = tmp_path / "normalize_text.yaml"
    normalize_config.write_text(_NORMALIZE, encoding="utf-8")
    clean_log = tmp_path / "custom" / "clean.log"
    normalize_log = tmp_path / "custom" / "normalize.log"

    clean_dataset(
        dataset_config,
        clean_config,
        output_dir=tmp_path / "interim",
        log_path=clean_log,
    )
    normalize_dataset(
        dataset_config,
        normalize_config,
        input_dir=tmp_path / "interim",
        output_dir=tmp_path / "processed",
        log_path=normalize_log,
    )

    assert "INFO" in clean_log.read_text(encoding="utf-8")
    assert "INFO" in normalize_log.read_text(encoding="utf-8")
    assert not (tmp_path / "artifacts").exists()


def test_clean_records_a_missing_source(tmp_path: Path) -> None:
    """A missing raw file is recorded at error level and still raised."""
    raw = tmp_path / "raw"
    raw.mkdir()
    dataset_config = tmp_path / "dataset.yaml"
    _dataset_config(dataset_config, raw, tmp_path / "cache")
    clean_config = tmp_path / "clean_text.yaml"
    clean_config.write_text(_CLEAN, encoding="utf-8")
    log_path = tmp_path / "clean.log"

    with pytest.raises(CleanTextError, match=r"valid\.jsonl"):
        clean_dataset(
            dataset_config,
            clean_config,
            output_dir=tmp_path / "interim",
            log_path=log_path,
        )

    log_text = log_path.read_text(encoding="utf-8")
    assert "ERROR" in log_text
    assert "Cleaning failed for valid.jsonl" in log_text


def test_normalize_records_a_missing_source(tmp_path: Path) -> None:
    """A missing cleaned file is recorded at error level and still raised."""
    raw = tmp_path / "raw"
    raw.mkdir()
    dataset_config = tmp_path / "dataset.yaml"
    _dataset_config(dataset_config, raw, tmp_path / "cache")
    normalize_config = tmp_path / "normalize_text.yaml"
    normalize_config.write_text(_NORMALIZE, encoding="utf-8")
    log_path = tmp_path / "normalize.log"

    with pytest.raises(NormalizeTextError, match=r"valid\.jsonl"):
        normalize_dataset(
            dataset_config,
            normalize_config,
            input_dir=tmp_path / "interim",
            output_dir=tmp_path / "processed",
            log_path=log_path,
        )

    log_text = log_path.read_text(encoding="utf-8")
    assert "ERROR" in log_text
    assert "Normalization failed for valid.jsonl" in log_text


def test_clean_and_normalize_record_a_disabled_save(tmp_path: Path) -> None:
    """Disabled saving is recorded and writes no derived files."""
    raw = tmp_path / "raw"
    raw.mkdir()
    dataset_config = tmp_path / "dataset.yaml"
    _dataset_config(dataset_config, raw, tmp_path / "cache")
    clean_config = tmp_path / "clean_text.yaml"
    clean_config.write_text(
        _CLEAN.replace("save_processed: true", "save_processed: false"),
        encoding="utf-8",
    )
    normalize_config = tmp_path / "normalize_text.yaml"
    normalize_config.write_text(
        _NORMALIZE.replace("save_processed: true", "save_processed: false"),
        encoding="utf-8",
    )
    clean_log = tmp_path / "clean.log"
    normalize_log = tmp_path / "normalize.log"

    cleaned = clean_dataset(
        dataset_config,
        clean_config,
        output_dir=tmp_path / "interim",
        log_path=clean_log,
    )
    written = normalize_dataset(
        dataset_config,
        normalize_config,
        input_dir=tmp_path / "interim",
        output_dir=tmp_path / "processed",
        log_path=normalize_log,
    )

    assert cleaned == ()
    assert written == ()
    assert "save_processed is false" in clean_log.read_text(encoding="utf-8")
    assert "save_processed is false" in normalize_log.read_text(encoding="utf-8")
    assert not (tmp_path / "interim").exists()
    assert not (tmp_path / "processed").exists()
