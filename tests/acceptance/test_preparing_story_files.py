"""Acceptance scenarios for rewriting story files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from tests.acceptance.story_text import story_body
from tiny_story.domain.errors import CleanTextError, NormalizeTextError
from tiny_story.infrastructure.text_processing.clean_text.jsonl_story_cleaner import (
    JsonlStoryCleaner,
)
from tiny_story.infrastructure.text_processing.normalize_text.jsonl_story_normalizer import (
    JsonlStoryNormalizer,
)

if TYPE_CHECKING:
    from pathlib import Path

scenarios("features/preparing_story_files.feature")

_PAGE_HEADER_PATTERN = r"^Page \| \d+ .*$"
_JUNK_LINE_PATTERN = r"^[/\\-]+$"
_SENTENCE_MARK = "|"


@dataclass(frozen=True, slots=True)
class RawStoryFile:
    """A raw JSONL story file and the bytes the scenario started with.

    Attributes:
        path: Raw file path. The file is absent when the scenario has no source.
        original: Original file text. Empty when the source is absent.
    """

    path: Path
    original: str


@dataclass(frozen=True, slots=True)
class DerivedStoryFile:
    """The derived JSONL path and any text already stored there.

    Attributes:
        path: Derived file path.
        prior_text: Text already in the derived file, when the scenario created one.
    """

    path: Path
    prior_text: str | None


@dataclass(frozen=True, slots=True)
class Preparation:
    """The derived path and any refusal from one rewrite.

    Attributes:
        derived: Derived file path.
        refusal: Refusal text. None when the rewrite finished.
    """

    derived: Path
    refusal: str | None


def file_text(docstring: str) -> str:
    """Return file text for a Gherkin docstring.

    Args:
        docstring: Docstring attached to a file step.

    Returns:
        The file text with one trailing newline.
    """
    return story_body(docstring) + "\n"


def clean_story_file(source: Path, destination: Path) -> str | None:
    """Clean a story file with the shipped rules.

    Args:
        source: Raw JSONL file.
        destination: Derived JSONL file.

    Returns:
        The refusal text, or None when the derived file was written.
    """
    cleaner = JsonlStoryCleaner(
        enabled=True,
        page_header_pattern=_PAGE_HEADER_PATTERN,
        junk_line_pattern=_JUNK_LINE_PATTERN,
        reflow_hard_wrapped_lines=True,
        join_hyphenated_line_breaks=True,
    )
    try:
        cleaner.clean_file(source, destination)
    except CleanTextError as exc:
        return str(exc)
    return None


def normalize_story_file(source: Path, destination: Path) -> str | None:
    """Normalize a story file with the shipped sentence mark.

    Args:
        source: JSONL file to read.
        destination: Derived JSONL file.

    Returns:
        The refusal text, or None when the derived file was written.
    """
    normalizer = JsonlStoryNormalizer(enabled=True, sentence_delimiter=_SENTENCE_MARK)
    try:
        normalizer.normalize_file(source, destination)
    except NormalizeTextError as exc:
        return str(exc)
    return None


@pytest.fixture
def derived_story_file(tmp_path: Path) -> DerivedStoryFile:
    """Return a derived path that the scenario has not created yet.

    Args:
        tmp_path: Per-test directory.

    Returns:
        A derived file that does not exist yet.
    """
    return DerivedStoryFile(path=tmp_path / "derived.jsonl", prior_text=None)


@given("a raw story file:", target_fixture="raw_story_file")
def raw_story_file(tmp_path: Path, docstring: str) -> RawStoryFile:
    """Write the raw story file for this scenario.

    Args:
        tmp_path: Per-test directory.
        docstring: JSONL contents from the scenario.

    Returns:
        The raw file and its original text.
    """
    original = file_text(docstring)
    path = tmp_path / "raw.jsonl"
    path.write_text(original, encoding="utf-8")
    return RawStoryFile(path=path, original=original)


@given("no raw story file", target_fixture="raw_story_file")
def no_raw_story_file(tmp_path: Path) -> RawStoryFile:
    """Point at a raw path that has no file.

    Args:
        tmp_path: Per-test directory.

    Returns:
        A missing raw file.
    """
    return RawStoryFile(path=tmp_path / "missing.jsonl", original="")


@given("the derived file is the raw story file", target_fixture="derived_story_file")
def derived_file_is_raw(raw_story_file: RawStoryFile) -> DerivedStoryFile:
    """Use the raw path as the derived path.

    Args:
        raw_story_file: Raw file for this scenario.

    Returns:
        A derived location that is the raw file.
    """
    return DerivedStoryFile(
        path=raw_story_file.path, prior_text=raw_story_file.original
    )


@given("an existing derived story file:", target_fixture="derived_story_file")
def existing_derived_story_file(tmp_path: Path, docstring: str) -> DerivedStoryFile:
    """Write a derived file that a failed rewrite must leave in place.

    Args:
        tmp_path: Per-test directory.
        docstring: Text already stored in the derived file.

    Returns:
        The existing derived file.
    """
    prior_text = file_text(docstring)
    path = tmp_path / "derived.jsonl"
    path.write_text(prior_text, encoding="utf-8")
    return DerivedStoryFile(path=path, prior_text=prior_text)


@when("the story file is cleaned", target_fixture="preparation")
def story_file_is_cleaned(
    raw_story_file: RawStoryFile,
    derived_story_file: DerivedStoryFile,
) -> Preparation:
    """Clean the raw story file into the derived path.

    Args:
        raw_story_file: Raw file for this scenario.
        derived_story_file: Derived path for this scenario.

    Returns:
        The derived path and any refusal.
    """
    refusal = clean_story_file(raw_story_file.path, derived_story_file.path)
    return Preparation(derived=derived_story_file.path, refusal=refusal)


@when("the story file is normalized", target_fixture="preparation")
def story_file_is_normalized(
    raw_story_file: RawStoryFile,
    derived_story_file: DerivedStoryFile,
) -> Preparation:
    """Normalize the raw story file into the derived path.

    Args:
        raw_story_file: Raw file for this scenario.
        derived_story_file: Derived path for this scenario.

    Returns:
        The derived path and any refusal.
    """
    refusal = normalize_story_file(raw_story_file.path, derived_story_file.path)
    return Preparation(derived=derived_story_file.path, refusal=refusal)


@then("the derived story file is:")
def derived_story_file_is(preparation: Preparation, docstring: str) -> None:
    """Assert the derived file text.

    Args:
        preparation: Rewrite result for this scenario.
        docstring: Expected JSONL contents.
    """
    assert preparation.refusal is None
    assert preparation.derived.read_text(encoding="utf-8") == file_text(docstring)


@then("the raw story file is unchanged")
def raw_story_file_is_unchanged(raw_story_file: RawStoryFile) -> None:
    """Assert the raw file still holds its original text.

    Args:
        raw_story_file: Raw file for this scenario.
    """
    assert raw_story_file.path.read_text(encoding="utf-8") == raw_story_file.original


@then(parsers.parse('preparation is refused because "{reason}"'))
def preparation_is_refused(preparation: Preparation, reason: str) -> None:
    """Assert the rewrite was refused for the stated reason.

    Args:
        preparation: Rewrite result for this scenario.
        reason: Text the refusal must contain.
    """
    assert preparation.refusal is not None
    assert reason in preparation.refusal


@then("no derived story file is written")
def no_derived_story_file_is_written(preparation: Preparation) -> None:
    """Assert the derived path was not created.

    Args:
        preparation: Rewrite result for this scenario.
    """
    assert not preparation.derived.exists()


@then("the derived story file is unchanged")
def derived_story_file_is_unchanged(
    preparation: Preparation,
    derived_story_file: DerivedStoryFile,
) -> None:
    """Assert a failed rewrite left the existing derived file in place.

    Args:
        preparation: Rewrite result for this scenario.
        derived_story_file: Derived file from scenario setup.
    """
    assert derived_story_file.prior_text is not None
    assert (
        preparation.derived.read_text(encoding="utf-8") == derived_story_file.prior_text
    )
