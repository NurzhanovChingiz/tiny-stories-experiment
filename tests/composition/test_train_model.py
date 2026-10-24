"""Behavior of configured Lightning training."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from tests.infrastructure.lightning.corpus import write_tokenized_corpus
from tiny_story.composition import train_model
from tiny_story.domain.errors import TrainError

if TYPE_CHECKING:
    from pathlib import Path

try:
    importlib.import_module("tiny_story.infrastructure.lightning.run")
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
enable_progress_bar: false
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


@pytest.fixture(autouse=True)
def _isolate_default_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the default train log in the test directory.

    Args:
        tmp_path: Per-test directory.
        monkeypatch: Pytest patch helper used to change the working directory.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)


def test_train_model_fits_a_tiny_corpus_and_writes_artifacts(tmp_path: Path) -> None:
    """One epoch on a fixture corpus writes checkpoints and a run log."""
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

    outcome = train_model(config_path)

    assert outcome.checkpoint_dir == checkpoint_dir
    assert outcome.log_dir == log_dir
    log_path = tmp_path / "artifacts" / "train" / "logs" / "other_corpus.log"
    assert "Wrote checkpoints" in log_path.read_text(encoding="utf-8")


def test_train_model_records_missing_tokenized_data(tmp_path: Path) -> None:
    """A missing tokenized directory is recorded and is not invented."""
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        _TRAIN_YAML.format(
            data_dir=tmp_path / "missing",
            checkpoint_dir=tmp_path / "checkpoints",
            log_dir=tmp_path / "lightning_logs",
        ),
        encoding="utf-8",
    )

    with pytest.raises(TrainError, match="does not exist"):
        train_model(config_path)
