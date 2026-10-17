"""Tokenize the text field of JSONL story files."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tiny_story.domain.errors import TokenizeTextError

if TYPE_CHECKING:
    from collections.abc import Iterable
from tiny_story.infrastructure.text_processing.tokenize_text.steps import (
    build_vocabulary,
    whitespace_tokens,
)
from tiny_story.infrastructure.text_processing.tokenize_text.whitespace_tokenizer import (
    WhitespaceTokenizer,
)


class JsonlStoryTokenizer:
    """Write token ids for each story and one shared vocabulary."""

    def __init__(self, *, enabled: bool) -> None:
        """Store whether stories are encoded.

        Args:
            enabled: Whether token ids are added. When false, stories are copied.

        Returns:
            None.
        """
        self._enabled = enabled

    def tokenize_files(
        self,
        sources: tuple[Path, ...],
        destinations: tuple[Path, ...],
        vocabulary_path: Path,
    ) -> None:
        """Write derived JSONL files and, when encoding, one vocabulary.

        Each source is read once. A failed rewrite removes its temporary file
        and leaves an existing destination in place. The vocabulary is built
        from every loaded story before the first derived file is published.

        Args:
            sources: Processed JSONL files. Each object has a string ``text``.
            destinations: Derived JSONL files, paired with ``sources``.
            vocabulary_path: JSON object of token to id. Written only when
                tokenization is enabled.

        Returns:
            None.

        Raises:
            TokenizeTextError: A source is missing, a destination would replace
                its source, the vocabulary would replace a source, a line is not
                a story object, or a derived file could not be written.

        Flow:
            1. Reject an unpaired list, a missing source, or a path that would overwrite a source.
            2. Copy the stories when tokenization is off.
            3. Load every story once, build one vocabulary, then publish each encoded file.
        """
        # 1. Reject an unpaired list, a missing source, or a path that would overwrite a source.
        _require_pairs(sources, destinations, vocabulary_path)

        # 2. Copy the stories when tokenization is off.
        if not self._enabled:
            for source, destination in zip(sources, destinations, strict=True):
                _publish_copy(source, destination)
            return

        # 3. Load every story once, build one vocabulary, then publish each encoded file.
        loaded = tuple(_load_stories(source) for source in sources)
        vocabulary = build_vocabulary(_tokens_from_stories(loaded))
        _publish_json(vocabulary_path, vocabulary)
        tokenizer = WhitespaceTokenizer(enabled=True, vocabulary=vocabulary)
        for source, stories, destination in zip(
            sources, loaded, destinations, strict=True
        ):
            _publish_stories(stories, destination, tokenizer, source.name)


def _require_pairs(
    sources: tuple[Path, ...],
    destinations: tuple[Path, ...],
    vocabulary_path: Path,
) -> None:
    """Reject a tokenization request that could destroy a source file.

    Args:
        sources: Processed JSONL files.
        destinations: Derived JSONL files.
        vocabulary_path: Vocabulary file.

    Returns:
        None.

    Raises:
        TokenizeTextError: The lists differ in length, a source is missing, or
            a destination or the vocabulary resolves to a source.
    """
    if len(sources) != len(destinations):
        message = "Tokenization sources and destinations must pair"
        raise TokenizeTextError(message)
    resolved_sources = set()
    for source in sources:
        if not source.is_file():
            message = f"Story file does not exist: {source}"
            raise TokenizeTextError(message)
        resolved_sources.add(source.resolve())
    for source, destination in zip(sources, destinations, strict=True):
        if source.resolve() == destination.resolve():
            message = f"Refusing to replace story file {source}"
            raise TokenizeTextError(message)
    if vocabulary_path.resolve() in resolved_sources:
        message = f"Refusing to replace story file {vocabulary_path}"
        raise TokenizeTextError(message)


def _load_stories(source: Path) -> tuple[tuple[dict[str, object], str], ...]:
    """Read one story file and parse each non-blank line once.

    Args:
        source: Processed JSONL file that has already been checked.

    Returns:
        Each story object paired with its string ``text``, in file order.
        Blank lines are omitted.

    Raises:
        TokenizeTextError: A line is not a story object, or the file could not
            be decoded.
    """
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        message = f"Could not tokenize {source.name}"
        raise TokenizeTextError(message) from exc
    return tuple(
        _story_record(line, line_number, source.name)
        for line_number, line in enumerate(lines, start=1)
        if line.strip() != ""
    )


def _story_record(
    line: str,
    line_number: int,
    filename: str,
) -> tuple[dict[str, object], str]:
    """Return one JSONL line as a story object and its string ``text``.

    Args:
        line: One non-empty JSONL line.
        line_number: 1-based line number used in errors.
        filename: Source name used when the line is not JSON.

    Returns:
        The story object and the text that will be tokenized.

    Raises:
        TokenizeTextError: The line is not an object with a string ``text``.
    """
    try:
        parsed: object = json.loads(line)
    except json.JSONDecodeError as exc:
        message = f"Could not tokenize {filename}"
        raise TokenizeTextError(message) from exc
    if not isinstance(parsed, dict):
        message = f"JSONL line {line_number} must be an object"
        raise TokenizeTextError(message)
    body = parsed.get("text")
    if not isinstance(body, str):
        message = f"JSONL line {line_number} must contain a string text field"
        raise TokenizeTextError(message)
    return parsed, body


def _tokens_from_stories(
    loaded: tuple[tuple[tuple[dict[str, object], str], ...], ...],
) -> Iterable[str]:
    """Yield whitespace tokens from stories already loaded in file order.

    Args:
        loaded: Story objects and text for each source, in source order.

    Returns:
        Tokens in file order, then story order.
    """
    for stories in loaded:
        for _story, text in stories:
            yield from whitespace_tokens(text)


def _publish_stories(
    stories: tuple[tuple[dict[str, object], str], ...],
    destination: Path,
    tokenizer: WhitespaceTokenizer,
    filename: str,
) -> None:
    """Write one derived file from stories that were already parsed.

    Args:
        stories: Story objects and text from one source, in file order.
        destination: Derived JSONL file.
        tokenizer: Encoder built from the shared vocabulary.
        filename: Source name used when the derived file cannot be published.

    Returns:
        None.

    Raises:
        TokenizeTextError: The derived file could not be written.
    """
    temporary = _write_temporary(
        destination,
        tuple(_encoded_story(story, text, tokenizer) for story, text in stories),
    )
    _replace(temporary, destination, filename)


def _encoded_story(
    story: dict[str, object],
    text: str,
    tokenizer: WhitespaceTokenizer,
) -> str:
    """Return one JSONL line with ``token_ids`` added to a loaded story.

    Args:
        story: Story object loaded from one source line.
        text: Story text already taken from ``story``.
        tokenizer: Encoder built from the shared vocabulary.

    Returns:
        The derived JSON object plus a newline.
    """
    ids, _result = tokenizer.tokenize(text)
    story["token_ids"] = list(ids)
    return json.dumps(story, ensure_ascii=False) + "\n"


def _publish_copy(source: Path, destination: Path) -> None:
    """Copy one story file without adding token ids.

    Args:
        source: Processed JSONL file.
        destination: Derived JSONL file.

    Returns:
        None.

    Raises:
        TokenizeTextError: The copy could not be written.
    """
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        message = f"Could not tokenize {source.name}"
        raise TokenizeTextError(message) from exc
    if text and not text.endswith("\n"):
        text += "\n"
    temporary = _write_temporary(destination, (text,) if text else ())
    _replace(temporary, destination, source.name)


def _publish_json(path: Path, document: dict[str, int]) -> None:
    """Write a JSON object by replacing ``path`` only after the write succeeds.

    Args:
        path: Final JSON file.
        document: Token to id.

    Returns:
        None.

    Raises:
        TokenizeTextError: The vocabulary file could not be written.
    """
    ordered = dict(sorted(document.items(), key=lambda item: item[1]))
    text = json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"
    temporary = _write_temporary(path, (text,))
    _replace(temporary, path, path.name)


def _write_temporary(destination: Path, lines: tuple[str, ...]) -> Path:
    """Write lines to an exclusive temporary file beside ``destination``.

    Args:
        destination: Final file.
        lines: Text to write, in order.

    Returns:
        The temporary path.

    Raises:
        TokenizeTextError: The temporary file could not be created.
    """
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(name)
        temporary.write_text("".join(lines), encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        message = f"Could not tokenize {destination.name}"
        raise TokenizeTextError(message) from exc
    return temporary


def _replace(temporary: Path, destination: Path, filename: str) -> None:
    """Publish a temporary file, or remove it when the publish fails.

    Args:
        temporary: Exclusive file that holds the derived bytes.
        destination: Final file.
        filename: Name used in the error.

    Returns:
        None.

    Raises:
        TokenizeTextError: The destination could not be replaced.
    """
    try:
        temporary.replace(destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        message = f"Could not tokenize {filename}"
        raise TokenizeTextError(message) from exc
