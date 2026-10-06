tiny-story-lm/
│
├── .gitignore
├── .python-version
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
├── README.md
│
├── config/
│   ├── global.yaml
│   ├── tokenizer.yaml
│   ├── model/
│   │   ├── 100k.yaml
│   │   ├── 500k.yaml
│   │   └── 1m.yaml
│   └── training/
│       ├── debug.yaml
│       ├── tiny.yaml
│       └── full.yaml
│
├── data/
│   ├── raw/
│   │   └── tiny_stories_raw/
│   ├── processed/
│   └── tokenized/
│
├── artifacts/
│   ├── runs/
│   ├── checkpoints/
│   ├── tokenizers/
│   └── logs/
│
├── docs/
│   ├── architecture.md
│   ├── training.md
│   ├── data_flow.md
│   └── experiments.md
│
├── notebooks/
│   ├── 00_dataset.ipynb
│   ├── 01_tokenizer.ipynb
│   ├── 02_model.ipynb
│   ├── 03_overfit_batch.ipynb
│   └── 04_generation.ipynb
│
├── scripts/
│   └── smoke_train.sh
│
├── tests/
│   ├── architecture/
│   │   └── test_import_rules.py
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   │   └── test_download_dataset.py
│   │   └── infrastructure/
│   │       └── test_huggingface_tinystories_repository.py
│   ├── integration/
│   │   ├── test_tinystories_repository.py
│   │   ├── test_tokenizer.py
│   │   └── test_lightning_trainer.py
│   └── acceptance/
│       └── test_train_and_generate.py
│
└── src/
    └── tinylm/
        │
        ├── __init__.py
        ├── composition.py
        │
        ├── domain/
        │   │
        │   ├── dataset/
        │   │   ├── dataset_spec.py
        │   │   └── text_sample.py
        │   │
        │   ├── modeling/
        │   │   ├── model_spec.py
        │   │   └── model_size.py
        │   │
        │   ├── tokenization/
        │   │   └── tokenizer_spec.py
        │   │
        │   ├── training/
        │   │   ├── training_run.py
        │   │   ├── training_run_id.py
        │   │   ├── training_status.py
        │   │   └── training_metrics.py
        │   │
        │   ├── generation/
        │   │   ├── generation_request.py
        │   │   └── generated_text.py
        │   │
        │   └── errors.py
        │
        ├── application/
        │   │
        │   ├── config/
        │   │   ├── dataset.py
        │   │   ├── tokenizer.py
        │   │   ├── model.py
        │   │   ├── optimizer.py
        │   │   └── training.py
        │   │
        │   ├── ports/
        │   │   ├── dataset_repository.py
        │   │   ├── published_dataset_source.py
        │   │   ├── raw_dataset_store.py
        │   │   ├── tokenizer_repository.py
        │   │   ├── tokenizer_trainer.py
        │   │   ├── model_trainer.py
        │   │   ├── checkpoint_repository.py
        │   │   ├── training_run_repository.py
        │   │   ├── metrics_logger.py
        │   │   └── text_generator.py
        │   │
        │   ├── use_cases/
        │   │   ├── download_dataset.py
        │   │   ├── prepare_dataset.py
        │   │   ├── train_tokenizer.py
        │   │   ├── train_model.py
        │   │   ├── evaluate_model.py
        │   │   └── generate_text.py
        │   │
        │   └── results/
        │       ├── download_outcome.py
        │       ├── training_outcome.py
        │       └── evaluation_outcome.py
        │
        ├── infrastructure/
        │   │
        │   ├── config/
        │   │   └── yaml_loader.py
        │   │
        │   ├── datasets/
        │   │   ├── huggingface_tinystories_repository.py
        │   │   └── filesystem_dataset_repository.py
        │   │
        │   ├── tokenization/
        │   │   ├── bpe_tokenizer_trainer.py
        │   │   └── filesystem_tokenizer_repository.py
        │   │
        │   ├── storage/
        │   │   ├── filesystem_checkpoint_repository.py
        │   │   └── filesystem_training_run_repository.py
        │   │
        │   └── lightning/
        │       │
        │       ├── models/
        │       │   ├── causal_lm.py
        │       │   ├── transformer_block.py
        │       │   ├── attention.py
        │       │   ├── mlp.py
        │       │   └── positional_embedding.py
        │       │
        │       ├── data/
        │       │   ├── causal_lm_dataset.py
        │       │   └── tinystories_datamodule.py
        │       │
        │       ├── training/
        │       │   ├── lightning_module.py
        │       │   ├── lightning_model_trainer.py
        │       │   ├── trainer_factory.py
        │       │   └── callbacks.py
        │       │
        │       └── inference/
        │           ├── checkpoint_loader.py
        │           └── autoregressive_generator.py
        │
        └── entrypoints/
            └── cli/
                ├── __main__.py
                └── commands/
                    ├── download.py
                    ├── train_tokenizer.py
                    ├── train.py
                    ├── evaluate.py
                    └── generate.py
