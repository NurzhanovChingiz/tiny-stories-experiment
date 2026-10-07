"""Byte-level BPE trainer that serializes a tokenizer document."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from tiny_stories_experiment.application.ports.tokenizer_trainer import TokenizerTrainer
from tiny_stories_experiment.application.results.trained_tokenizer import (
    TrainedTokenizer,
)
from tiny_stories_experiment.domain.errors import TokenizerTrainingError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from tiny_stories_experiment.domain.tokenization.tokenizer_spec import (
        TokenizerSpec,
    )

_BYTE_ALPHABET_SIZE = 256


class BpeTokenizerTrainer(TokenizerTrainer):
    """Train a byte-level BPE tokenizer and return its JSON document.

    Every byte is in the initial alphabet, so later encoding can represent any
    Unicode story. Special tokens are reserved first. Merges fill the remaining
    vocab until the spec's vocab size or the corpus runs out of frequent pairs.
    """

    def train(self, spec: TokenizerSpec, texts: Iterable[str]) -> TrainedTokenizer:
        """Train on ``texts`` and return the serialized tokenizer.

        Args:
            spec: Vocab size, minimum pair frequency, and special tokens.
            texts: Story text in read order. The trainer consumes this once.

        Returns:
            The serialized tokenizer and how many stories were read.

        Raises:
            TokenizerTrainingError: ``vocab_size`` is below the byte alphabet
                plus the special tokens, or the stream contains no stories.

        Flow:
            1. Vocab floor — refuse a size below the byte alphabet and specials.
            2. Fit merges — train byte-level BPE on the story stream.
            3. Empty corpus — refuse a run that read no stories.
            4. JSON document — return the serialized vocabulary and story count.
        """
        # 1. Vocab floor
        _require_vocab_size(spec)

        # 2. Fit merges
        counter = _StoryCounter()
        tokenizer = _fit(spec, counter.observe(texts))

        # 3. Empty corpus
        if counter.count == 0:
            message = "Prepared splits contain no stories."
            raise TokenizerTrainingError(message)

        # 4. JSON document
        return TrainedTokenizer(
            content=_json_bytes(tokenizer),
            vocab_size=_vocab_size(tokenizer),
            story_count=counter.count,
        )


class _StoryCounter:
    """Count stories as a trainer reads them."""

    def __init__(self) -> None:
        """Start from zero stories."""
        self.count = 0

    def observe(self, texts: Iterable[str]) -> Iterator[str]:
        """Yield each story and count it.

        Args:
            texts: Story text in read order.

        Returns:
            The same stories, in the same order.
        """
        for text in texts:
            self.count += 1
            yield text


def _require_vocab_size(spec: TokenizerSpec) -> None:
    """Refuse a vocab size that cannot hold the byte alphabet and special tokens.

    Args:
        spec: Vocab size and special tokens to check.

    Raises:
        TokenizerTrainingError: ``vocab_size`` is below that floor.
    """
    minimum = _BYTE_ALPHABET_SIZE + len(spec.special_tokens)
    if spec.vocab_size < minimum:
        message = (
            f"Tokenizer vocab size {spec.vocab_size} is below the byte-level "
            f"minimum of {minimum}."
        )
        raise TokenizerTrainingError(message)


def _fit(spec: TokenizerSpec, texts: Iterable[str]) -> Tokenizer:
    """Fit a byte-level BPE model on ``texts``.

    Args:
        spec: Vocab size, minimum pair frequency, and special tokens.
        texts: Story text in read order.

    Returns:
        The trained tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=spec.vocab_size,
        min_frequency=spec.min_frequency,
        show_progress=False,
        special_tokens=list(spec.special_tokens),
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def _json_bytes(tokenizer: Tokenizer) -> bytes:
    """Return the tokenizer JSON as UTF-8 bytes.

    Args:
        tokenizer: Trained tokenizer to serialize.

    Returns:
        The UTF-8 JSON document.

    Raises:
        TokenizerTrainingError: The library did not return a JSON string.
    """
    document = tokenizer.to_str(pretty=False)
    if not isinstance(document, str):
        message = "Trained tokenizer did not serialize to text."
        raise TokenizerTrainingError(message)
    return document.encode("utf-8")


def _vocab_size(tokenizer: Tokenizer) -> int:
    """Return the trained vocabulary size.

    Args:
        tokenizer: Trained tokenizer to measure.

    Returns:
        The number of tokens, including special tokens.

    Raises:
        TokenizerTrainingError: The library did not report an integer size.
    """
    size = tokenizer.get_vocab_size()
    if isinstance(size, bool) or not isinstance(size, int):
        message = "Trained tokenizer did not report an integer vocab size."
        raise TokenizerTrainingError(message)
    return size
