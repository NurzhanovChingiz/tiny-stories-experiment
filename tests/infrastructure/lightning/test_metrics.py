"""Tests for next-token metric accumulation."""

from __future__ import annotations

import pytest

try:
    import torch

    from tiny_story.infrastructure.lightning.metrics import LanguageModelMetrics
except ImportError:
    pytest.skip("torch is required", allow_module_level=True)


def test_metrics_ignore_pad_and_score_top1() -> None:
    """Pad targets are ignored and an exact top-1 hit is counted."""
    tracker = LanguageModelMetrics(vocab_size=4)
    logits = torch.tensor(
        [[[0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[1, 0]], dtype=torch.long)
    loss = torch.tensor(0.5)

    tracker.update(logits, targets, loss)
    scores = tracker.compute()

    assert scores["loss"] == pytest.approx(0.5)
    assert scores["acc_top1"] == pytest.approx(1.0)
    assert scores["acc_top5"] == pytest.approx(1.0)
    tracker.reset()
    assert tracker.compute()["loss"] == 0.0
