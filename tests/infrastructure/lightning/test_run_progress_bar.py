"""Lightning training forwards the progress-bar flag from configuration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests.infrastructure.lightning.corpus import write_tokenized_corpus
from tiny_story.infrastructure.config.train_config import load_train_config

if TYPE_CHECKING:
    from pathlib import Path

try:
    from tiny_story.infrastructure.lightning.run import LightningTrainingRun
except ImportError:
    pytest.skip("torch and lightning are required", allow_module_level=True)


_TRAIN_YAML = """\
name: other_corpus
seed: 3
data_dir: {data_dir}
vocab_file: vocab.json
sequence_length: 8
batch_size: 2
epochs: 1
learning_rate: 0.05
weight_decay: 0.0
scheduler_type: cosine
warmup_epochs: 0
eta_min: 1.0e-6
step_size: 10
gamma: 0.1
max_lr: 0.05
min_delta: 0.001
patience: 2
num_workers: 0
pin_memory: false
persistent_workers: false
amp: false
precision: 32-true
enable_progress_bar: true
model:
  embedding_dim: 16
  hidden_size: 32
  num_layers: 1
  dropout: 0.0
split:
  test_size: 0.2
  val_size: 0.2
checkpoint_dir: {checkpoint_dir}
log_dir: {log_dir}
"""


def test_fit_forwards_enable_progress_bar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trainer is built with the configured progress-bar flag."""
    data_dir = tmp_path / "tokenized"
    write_tokenized_corpus(data_dir, stories=8)
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "lightning_logs"
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        _TRAIN_YAML.format(
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        ),
        encoding="utf-8",
    )
    config = load_train_config(config_path)
    captured: dict[str, object] = {}

    class _FakeTrainer:
        checkpoint_callback = SimpleNamespace(best_model_path="")

        def fit(self, *_args: object, **_kwargs: object) -> None:
            return None

    def _fake_build_trainer(**kwargs: object) -> _FakeTrainer:
        captured.update(kwargs)
        return _FakeTrainer()

    monkeypatch.setattr(
        "tiny_story.infrastructure.lightning.run.build_trainer",
        _fake_build_trainer,
    )

    LightningTrainingRun(config, data_dir, checkpoint_dir, log_dir).fit()

    assert captured["enable_progress_bar"] is True
