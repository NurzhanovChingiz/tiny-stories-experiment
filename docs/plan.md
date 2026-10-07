# Plan status

Status date: 2026-10-07. Markers below are a file inventory of this checkout. Unit tests, architecture tests, Ruff, and mypy were run for the tokenizer slice. The GPU smoke test was not re-run.

Package in code is `tiny_stories_experiment` under `src/tiny_stories_experiment/`. The sketch name `tinylm` is not a second tree.

Markers:

- `[done]` present and used by `make download` or `make prepare`, or by the ROCm container files
- `[partial]` file or target exists; planned behavior is not implemented
- `[next]` first unfinished vertical slice
- `[later]` not started

## Done

Download and prepare are the finished data path.

- `make download` runs `python -m tiny_stories_experiment.entrypoints.cli download`.
- It fetches `noanabeshima/TinyStoriesV2` files `TinyStoriesV2-GPT4-train.jsonl` and `TinyStoriesV2-GPT4-valid.jsonl` into `data/raw/tiny_stories_raw/`.
- A file that already matches the published byte count is left unchanged. A size mismatch stops the command. A finished download with the wrong size is removed.
- `make prepare` runs the `prepare` command.
- It reads those raw JSONL files and writes `data/processed/tiny_stories/train.jsonl` and `data/processed/tiny_stories/valid.jsonl`.
- Derived files are replaced only after every split is staged and the raw byte counts still match. Raw files stay immutable.

Code that implements that path:

- Domain: `src/tiny_stories_experiment/domain/dataset/dataset_spec.py`, `text_sample.py`, `domain/tokenization/tokenizer_spec.py`, `domain/errors.py`
- Use cases: `application/use_cases/download_dataset.py`, `prepare_dataset.py`, `train_tokenizer.py`
- Results: `application/results/download_outcome.py`, `prepare_outcome.py`, `trained_tokenizer.py`, `train_tokenizer_outcome.py`
- Ports: `published_dataset_source.py`, `raw_dataset_store.py`, `raw_story_source.py`, `processed_text_store.py`, `directory_file_size.py`, `tokenizer_trainer.py`, `tokenizer_repository.py`
- Infrastructure: `huggingface_tinystories_repository.py`, `filesystem_dataset_repository.py`, `jsonl_story_source.py`, `filesystem_processed_text_store.py`, `tokenization/bpe_tokenizer_trainer.py`, `tokenization/filesystem_tokenizer_repository.py`
- Wiring: `composition.py`
- CLI: `entrypoints/cli/__main__.py`, `commands/download.py`, `commands/prepare.py`, `commands/train_tokenizer.py`
- Tests: `tests/unit/application/test_download_dataset.py`, `test_prepare_dataset.py`, `test_train_tokenizer.py`, `tests/unit/domain/test_text_sample.py`, `test_tokenizer_spec.py`, `tests/unit/infrastructure/test_huggingface_tinystories_repository.py`, `tests/unit/entrypoints/cli/test_download_command.py`, `test_prepare_command.py`, `test_train_tokenizer_command.py`, `tests/architecture/test_import_rules.py`
- Operator notes: `RUNBOOK.md` (download, prepare, and train-tokenizer)
- Project files: `pyproject.toml`, `uv.lock`, `.python-version`, `.pre-commit-config.yaml`, `.gitignore`, `Makefile`

ROCm training container files are present. `make train` starts Jupyter Lab in that container. It does not train a language model. `make train-tokenizer` trains a byte-level BPE tokenizer on the host from the prepared splits and writes `artifacts/tokenizers/tiny_stories_bpe/tokenizer.json`. Contract tests for the GPU probe live in `tests/test_training_smoke.py` and `docker/training/smoke.py`. Container files: `docker/training/Dockerfile`, `docker-compose.yaml`, `.env.example`, `download_bigfiles.sh`. Notebook on disk: `notebooks/00_/gpu_pytorch.ipynb`.

## Files in docs

| File | Role |
| --- | --- |
| `docs/plan.md` | This status, done work, and next steps |
| `docs/rocm-training-docker-env.md` | Host and container contract for the ROCm Jupyter image |

Not in the tree yet: `docs/architecture.md`, `docs/training.md`, `docs/data_flow.md`, `docs/experiments.md`. `RUNBOOK.md` at the repo root covers download, prepare, and train-tokenizer.

## Next steps

The tokenizer slice is in place. Immediate step: small causal language model and an overfit-one-batch check.

1. **Add the small causal language model and an overfit-one-batch check.** Reason: the plan’s model and `03_overfit_batch` notebook have no modules yet. Success: one fixed batch reaches near-zero training loss in a test, with model code under `infrastructure/lightning/`.
2. **Point `make train` at a real training entrypoint inside the existing ROCm container.** Reason: `make train` currently starts Jupyter only. Success: one short training run writes a checkpoint under the container checkpoint volume and records train loss.
3. **Add evaluate and generate commands.** Reason: the CLI stops at `train-tokenizer`. Success: generate prints text from a saved checkpoint using the saved tokenizer.

Stay in `src/tiny_stories_experiment/`. Do not add a parallel `tinylm` package.

## Limitations

- `data/` is gitignored. This update did not train on the full prepared corpus.
- `docs/rocm-training-docker-env.md` records a GPU smoke and one Lightning epoch on this machine on 2026-08-27 in the original environment. This checkout has no Lightning training modules.
- Unit tests, architecture tests, Ruff, and mypy passed for the tokenizer slice. The GPU smoke test was not re-run.

## Target tree

```text
tiny-stories-experiment/
│
├── .gitignore                          [done]
├── .python-version                     [done]
├── .pre-commit-config.yaml             [done]
├── pyproject.toml                      [done]
├── uv.lock                             [done]
├── Makefile                            [partial] download, prepare, and train-tokenizer done; train starts Jupyter
├── RUNBOOK.md                          [partial] download, prepare, and train-tokenizer
├── README.md                           [later]
│
├── config/                             [later]
│   ├── global.yaml
│   ├── tokenizer.yaml                  [later] waits on the yaml loader
│   ├── model/
│   │   ├── 100k.yaml
│   │   ├── 500k.yaml
│   │   └── 1m.yaml
│   └── training/
│       ├── debug.yaml
│       ├── tiny.yaml
│       └── full.yaml
│
├── data/                               [partial] paths used; directory is gitignored
│   ├── raw/tiny_stories_raw/           [done] download destination
│   ├── processed/tiny_stories/         [done] prepare destination
│   └── tokenized/                      [later]
│
├── artifacts/                          [partial] tokenizer directory is the train-tokenizer destination
│   ├── runs/
│   ├── checkpoints/
│   ├── tokenizers/                     [done] train-tokenizer destination
│   └── logs/
│
├── docs/
│   ├── plan.md                         [done] this file
│   ├── rocm-training-docker-env.md     [done]
│   ├── architecture.md                 [later]
│   ├── training.md                     [later]
│   ├── data_flow.md                    [later]
│   └── experiments.md                  [later]
│
├── docker/training/                    [partial] ROCm Jupyter image; no model training
│
├── notebooks/
│   ├── 00_/gpu_pytorch.ipynb           [partial] GPU notebook, not the dataset notebook
│   ├── 00_dataset.ipynb                [later]
│   ├── 01_tokenizer.ipynb              [later]
│   ├── 02_model.ipynb                  [later]
│   ├── 03_overfit_batch.ipynb          [later]
│   └── 04_generation.ipynb             [later]
│
├── scripts/smoke_train.sh              [later]
│
├── tests/                              [partial] data path, tokenizer, and GPU-probe contract
│   ├── architecture/test_import_rules.py   [done]
│   ├── unit/domain/                        [partial] text sample and tokenizer spec
│   ├── unit/application/                   [partial] download, prepare, and train-tokenizer
│   ├── unit/infrastructure/                [partial] Hugging Face repository
│   ├── unit/entrypoints/cli/               [partial] download, prepare, and train-tokenizer
│   ├── test_training_smoke.py             [partial] probe contract, not a training run
│   ├── integration/                        [later]
│   └── acceptance/test_train_and_generate.py [later]
│
└── src/tiny_stories_experiment/
    ├── composition.py                  [partial] download, prepare, and train-tokenizer wiring
    ├── domain/
    │   ├── dataset/                    [done]
    │   ├── errors.py                   [partial] dataset and tokenizer errors
    │   ├── modeling/                   [later]
    │   ├── tokenization/               [done]
    │   ├── training/                   [later]
    │   └── generation/                 [later]
    ├── application/
    │   ├── config/                     [later]
    │   ├── ports/                      [partial] dataset and tokenizer ports
    │   ├── use_cases/
    │   │   ├── download_dataset.py     [done]
    │   │   ├── prepare_dataset.py      [done]
    │   │   ├── train_tokenizer.py      [done]
    │   │   ├── train_model.py          [later]
    │   │   ├── evaluate_model.py       [later]
    │   │   └── generate_text.py        [later]
    │   └── results/                    [partial] download, prepare, and tokenizer outcomes
    ├── infrastructure/
    │   ├── config/yaml_loader.py       [later]
    │   ├── datasets/                   [done]
    │   ├── tokenization/               [done]
    │   ├── storage/                    [later]
    │   └── lightning/                  [later]
    └── entrypoints/cli/commands/
        ├── download.py                 [done]
        ├── prepare.py                  [done]
        ├── train_tokenizer.py          [done]
        ├── train.py                    [later]
        ├── evaluate.py                 [later]
        └── generate.py                 [later]
```
