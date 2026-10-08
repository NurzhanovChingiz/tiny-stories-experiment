"""CPU proof that the tiny causal LM can memorize one fixed batch."""

from __future__ import annotations

import logging
import math
import time

import pytest
import torch

from tiny_stories_experiment.application.ports.causal_language_model import (
    CausalLanguageModel,
)
from tiny_stories_experiment.domain.errors import InvalidTokenBatchError
from tiny_stories_experiment.domain.modeling.model_spec import ModelSpec
from tiny_stories_experiment.infrastructure.lightning.causal_lm import (
    FIXED_OVERFIT_TOKEN_IDS,
    NEAR_ZERO_TRAINING_LOSS,
    OVERFIT_STEP_COUNT,
    LightningCausalLanguageModel,
    build_tiny_debug_model,
    fit_token_batch,
)

_PARAMETER_LIMIT = 500_000
_WALL_CLOCK_LIMIT_SECONDS = 120.0
_LOGGER = logging.getLogger(__name__)


def test_tiny_debug_model_stays_within_a_few_hundred_thousand_parameters() -> None:
    """The default debug model is the overfit-sized network, not a full LM."""
    model = build_tiny_debug_model()
    count = sum(parameter.numel() for parameter in model.parameters())

    assert 0 < count <= _PARAMETER_LIMIT


def test_forward_on_the_fixed_batch_is_finite() -> None:
    """One forward of the fixed integer batch yields a finite scalar loss."""
    model = build_tiny_debug_model()

    loss = model.loss_on_tokens(FIXED_OVERFIT_TOKEN_IDS)

    assert isinstance(model, CausalLanguageModel)
    assert isinstance(model, LightningCausalLanguageModel)
    assert model.spec == ModelSpec.tiny_debug()
    assert math.isfinite(loss)


def test_earlier_positions_ignore_a_later_token() -> None:
    """A causal mask keeps an edited final token out of earlier logits."""
    model = build_tiny_debug_model()
    original = ((1, 2, 3, 4, 5, 6, 7, 8),)
    edited = ((1, 2, 3, 4, 5, 6, 7, 9),)

    original_logits = model.token_logits(original)
    edited_logits = model.token_logits(edited)

    assert torch.equal(original_logits[:, :-1, :], edited_logits[:, :-1, :])
    assert not torch.equal(original_logits[:, -1, :], edited_logits[:, -1, :])


def test_lightning_training_step_returns_a_finite_loss() -> None:
    """The Lightning hook scores the same fixed batch as a tensor."""
    model = build_tiny_debug_model()
    batch = torch.tensor(FIXED_OVERFIT_TOKEN_IDS)

    loss = model.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_loss_on_tokens_rejects_a_ragged_batch() -> None:
    """The adapter reports the domain batch error instead of scoring junk."""
    model = build_tiny_debug_model()

    with pytest.raises(InvalidTokenBatchError, match="same length"):
        model.loss_on_tokens(((1, 2, 3, 4), (1, 2)))


def test_overfit_one_batch_reaches_near_zero_loss() -> None:
    """Repeated steps on one fixed batch drive training loss under 0.05."""
    assert NEAR_ZERO_TRAINING_LOSS == 0.05
    model = build_tiny_debug_model()
    started = time.perf_counter()

    final_loss = fit_token_batch(
        model,
        FIXED_OVERFIT_TOKEN_IDS,
        OVERFIT_STEP_COUNT,
    )
    elapsed = time.perf_counter() - started

    _LOGGER.info("final_loss=%s duration_s=%.3f", final_loss, elapsed)
    assert final_loss < 0.05
    assert elapsed <= _WALL_CLOCK_LIMIT_SECONDS
