"""Behavior of dataset configuration parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tiny_story.application.use_cases.download_dataset import ExistingFilePolicy
from tiny_story.domain.errors import InvalidDatasetConfigError
from tiny_story.infrastructure.config.dataset_config import (
    DatasetConfig,
    HuggingFaceSourceConfig,
    load_dataset_config,
    parse_dataset_config,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _parse(text: str) -> DatasetConfig:
    """Parse a YAML document.

    Args:
        text: YAML mapping text.

    Returns:
        The parsed dataset configuration.

    Raises:
        InvalidDatasetConfigError: The document is not a YAML mapping.
    """
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        message = "Dataset configuration must be a YAML mapping"
        raise InvalidDatasetConfigError(message)
    return parse_dataset_config(loaded)


def test_default_config_selects_tinystories_v2() -> None:
    """The shipped configuration still acquires TinyStoriesV2."""
    config = load_dataset_config(_REPOSITORY_ROOT / "config/dataset.yaml")

    assert config.spec.name == "tiny_stories"
    assert config.spec.filenames == (
        "TinyStoriesV2-GPT4-train.jsonl",
        "TinyStoriesV2-GPT4-valid.jsonl",
    )
    assert config.destination == Path("data/raw/tiny_stories_raw")
    assert config.existing is ExistingFilePolicy.KEEP
    assert config.source == HuggingFaceSourceConfig(
        repository="noanabeshima/TinyStoriesV2",
        revision="main",
        cache_dir=Path("data/interim/huggingface"),
    )


def test_parse_huggingface_source_uses_defaults_for_optional_fields() -> None:
    """Revision and cache directory have defaults."""
    config = _parse(
        """
dataset:
  name: other_corpus
  files:
    - notes.txt
  destination: data/raw/other
  source:
    repository: some-org/other-corpus
""",
    )

    assert config.spec.name == "other_corpus"
    assert config.spec.filenames == ("notes.txt",)
    assert config.source == HuggingFaceSourceConfig(
        repository="some-org/other-corpus",
        revision="main",
        cache_dir=Path("data/interim/huggingface"),
    )


def test_parse_huggingface_source_keeps_explicit_fields() -> None:
    """Another Hub dataset keeps its repository, revision, cache, and overwrite."""
    config = _parse(
        """
dataset:
  name: other_corpus
  files:
    - train.txt
    - valid.txt
  destination: data/raw/other
  overwrite: true
  source:
    repository: some-org/other-corpus
    revision: abc123
    cache_dir: data/interim/custom
""",
    )

    assert config.spec.filenames == ("train.txt", "valid.txt")
    assert config.destination == Path("data/raw/other")
    assert config.existing is ExistingFilePolicy.REPLACE
    assert config.source == HuggingFaceSourceConfig(
        repository="some-org/other-corpus",
        revision="abc123",
        cache_dir=Path("data/interim/custom"),
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("dataset: []\n", "dataset must be a mapping"),
        (
            """
dataset:
  name: corpus
  files: []
  destination: data/raw/corpus
  source:
    repository: some-org/corpus
""",
            "dataset.files must list",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - ../escape.txt
  destination: data/raw/corpus
  source:
    repository: some-org/corpus
""",
            "single file name",
        ),
        (
            """
dataset:
  name: " "
  files:
    - notes.txt
  destination: data/raw/corpus
  source:
    repository: some-org/corpus
""",
            "dataset.name",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - notes.txt
  destination: " "
  source:
    repository: some-org/corpus
""",
            "dataset.destination",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - notes.txt
  destination: data/raw/corpus
  overwrite: 1
  source:
    repository: some-org/corpus
""",
            "dataset.overwrite",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - notes.txt
  destination: data/raw/corpus
  source: repository
""",
            "dataset.source must be a mapping",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - notes.txt
  destination: data/raw/corpus
  source:
    repository: TinyStories
""",
            "owner/name",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - notes.txt
  destination: data/raw/corpus
  source:
    repository: owner/name/extra
""",
            "owner/name",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - notes.txt
  destination: data/raw/corpus
  source:
    revision: main
""",
            "dataset.source.repository",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - notes.txt
  destination: data/raw/corpus
  source:
    repository: some-org/corpus
    revision: " "
""",
            "dataset.source.revision",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - notes.txt
  destination: data/raw/corpus
  source:
    repository: some-org/corpus
    cache_dir: ""
""",
            "dataset.source.cache_dir",
        ),
        (
            """
dataset:
  name: corpus
  files:
    - 1
  destination: data/raw/corpus
  source:
    repository: some-org/corpus
""",
            "dataset.files must list",
        ),
    ],
)
def test_parse_rejects_invalid_configuration(text: str, match: str) -> None:
    """Invalid documents name the field that must be fixed."""
    with pytest.raises(InvalidDatasetConfigError, match=match):
        _parse(text)


def test_load_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing configuration path fails before acquisition."""
    with pytest.raises(InvalidDatasetConfigError, match="does not exist"):
        load_dataset_config(tmp_path / "missing.yaml")


def test_load_rejects_invalid_yaml(tmp_path: Path) -> None:
    """A broken YAML document is reported as invalid configuration."""
    path = tmp_path / "dataset.yaml"
    path.write_text("dataset: [\n", encoding="utf-8")

    with pytest.raises(InvalidDatasetConfigError, match="not valid YAML"):
        load_dataset_config(path)


def test_load_rejects_yaml_that_is_not_a_mapping(tmp_path: Path) -> None:
    """A YAML list is not a dataset configuration."""
    path = tmp_path / "dataset.yaml"
    path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(InvalidDatasetConfigError, match="YAML mapping"):
        load_dataset_config(path)
