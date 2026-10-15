"""Behavior of the YAML configuration repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tiny_story.infrastructure.config.yaml_config_repository import (
    YamlConfigRepository,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SampleConfig:
    """Configuration used to observe repository results."""

    name: str
    count: int


def parse_sample(raw: dict[str, object]) -> SampleConfig:
    """Build a sample configuration from a YAML mapping.

    Args:
        raw: Mapping loaded from YAML.

    Returns:
        The sample configuration.
    """
    name = raw["name"]
    count = raw["count"]
    if not isinstance(name, str) or not isinstance(count, int):
        message = "Sample YAML values have unexpected types"
        raise TypeError(message)
    return SampleConfig(name=name, count=count)


def test_get_returns_the_parsed_yaml_mapping(tmp_path: Path) -> None:
    """A mapping document is parsed into the requested configuration type."""
    path = tmp_path / "sample.yaml"
    path.write_text("name: ada\ncount: 2\n", encoding="utf-8")
    repository = YamlConfigRepository(
        {SampleConfig: path},
        {SampleConfig: parse_sample},
    )

    result = repository.get(SampleConfig)

    assert result == SampleConfig(name="ada", count=2)


@pytest.mark.parametrize("text", ["- item\n", "hello\n", "5\n"])
def test_get_rejects_yaml_that_is_not_a_mapping(tmp_path: Path, text: str) -> None:
    """Documents other than a YAML mapping are rejected."""
    path = tmp_path / "sample.yaml"
    path.write_text(text, encoding="utf-8")
    repository = YamlConfigRepository(
        {SampleConfig: path},
        {SampleConfig: parse_sample},
    )

    with pytest.raises(TypeError, match="Expected YAML mapping"):
        repository.get(SampleConfig)


def test_get_requires_a_path_for_the_requested_type() -> None:
    """A type with no configured file is not loaded."""
    repository = YamlConfigRepository({}, {})

    with pytest.raises(KeyError, match="SampleConfig"):
        repository.get(SampleConfig)


def test_repositories_read_only_their_own_files(tmp_path: Path) -> None:
    """Each repository loads the file stored on that instance."""
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text("name: first\ncount: 1\n", encoding="utf-8")
    second_path.write_text("name: second\ncount: 2\n", encoding="utf-8")
    first = YamlConfigRepository(
        {SampleConfig: first_path},
        {SampleConfig: parse_sample},
    )
    second = YamlConfigRepository(
        {SampleConfig: second_path},
        {SampleConfig: parse_sample},
    )

    assert first.get(SampleConfig) == SampleConfig(name="first", count=1)
    assert second.get(SampleConfig) == SampleConfig(name="second", count=2)
