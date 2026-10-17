# Architecture map

Auto-generated path-only tree. Do not edit by hand.

Root: `tiny-stories-experiment/`

```
tiny-stories-experiment/
├── config/
│   ├── clean_text.yaml
│   ├── dataset.yaml
│   ├── normalize_text.yaml
│   └── tokenize.yaml
├── docker/
│   └── kernel/
│       ├── docker-compose.yaml
│       ├── Dockerfile
│       └── download_bigfiles.sh
├── docs/
│   ├── architecture-map.meta.json
│   └── architecture_map.md
├── notebooks/
│   ├── 00_environment/
│   │   └── gpu_pytorch.ipynb
│   └── check.ipynb
├── output/
│   ├── agent-performance-optimizer-skill/
│   │   ├── examples/
│   │   │   └── .gitkeep
│   │   ├── intermediate/
│   │   │   ├── admission_records.json
│   │   │   ├── admitted_general.json
│   │   │   ├── admitted_scoped.json
│   │   │   ├── contradiction_report.json
│   │   │   ├── contrast_records.json
│   │   │   ├── evidence_matrix.json
│   │   │   ├── excluded_candidates.json
│   │   │   ├── exemplar_discovery.json
│   │   │   ├── intake_assessment.json
│   │   │   ├── lens_analysis.md
│   │   │   ├── merge-summary.md
│   │   │   ├── merge_report.json
│   │   │   ├── open_world_task.json
│   │   │   ├── package_plan.json
│   │   │   └── research_plan.json
│   │   ├── references/
│   │   │   ├── .gitkeep
│   │   │   ├── R01-workload-binding.md
│   │   │   ├── R02-dependence-shared-state.md
│   │   │   ├── R03-bounded-concurrency.md
│   │   │   ├── R04-failure-cancellation.md
│   │   │   ├── R05-agent-quality.md
│   │   │   ├── R06-keep-or-revert.md
│   │   │   └── R07-simpler-change-first.md
│   │   ├── README.md
│   │   └── SKILL.md.draft
│   └── graph-reasoning-engine/
│       ├── agent-performance-optimizer-graph.json
│       ├── clean-docker-audit-graph.json
│       └── tokenize-next-step.json
├── src/
│   └── tiny_story/
│       ├── application/
│       │   ├── ports/
│       │   │   ├── __init__.py
│       │   │   ├── config_repository.py
│       │   │   ├── dataset_source.py
│       │   │   ├── file_reader.py
│       │   │   ├── prediction_writer.py
│       │   │   ├── raw_dataset_store.py
│       │   │   ├── run_logger.py
│       │   │   ├── text_cleaner.py
│       │   │   ├── text_normalizer.py
│       │   │   ├── text_source.py
│       │   │   └── text_tokenizer.py
│       │   ├── results/
│       │   │   ├── __init__.py
│       │   │   └── download_outcome.py
│       │   ├── use_cases/
│       │   │   ├── __init__.py
│       │   │   └── download_dataset.py
│       │   ├── __init__.py
│       │   └── existing_file_policy.py
│       ├── domain/
│       │   ├── entities/
│       │   │   ├── __init__.py
│       │   │   ├── cleaning.py
│       │   │   ├── dataset_spec.py
│       │   │   ├── normalization.py
│       │   │   ├── prediction.py
│       │   │   ├── text.py
│       │   │   └── tokenization.py
│       │   ├── __init__.py
│       │   └── errors.py
│       ├── entrypoints/
│       │   ├── cli/
│       │   │   ├── __init__.py
│       │   │   ├── __main__.py
│       │   │   ├── app.py
│       │   │   ├── clean.py
│       │   │   ├── download.py
│       │   │   ├── normalize.py
│       │   │   └── tokenize.py
│       │   └── __init__.py
│       ├── infrastructure/
│       │   ├── config/
│       │   │   ├── __init__.py
│       │   │   ├── clean_text_config.py
│       │   │   ├── dataset_config.py
│       │   │   ├── normalize_text_config.py
│       │   │   ├── tokenize_config.py
│       │   │   └── yaml_config_repository.py
│       │   ├── datasets/
│       │   │   ├── __init__.py
│       │   │   ├── filesystem_dataset_repository.py
│       │   │   └── huggingface_dataset_source.py
│       │   ├── io/
│       │   │   ├── __init__.py
│       │   │   └── read_text_jsonl.py
│       │   ├── logging/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_run_logger.py
│       │   ├── readers/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_file_reader.py
│       │   ├── runs/
│       │   │   ├── __init__.py
│       │   │   ├── download_log_layout.py
│       │   │   └── inference_run_layout.py
│       │   ├── text_processing/
│       │   │   ├── clean_text/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── compiled_patterns.py
│       │   │   │   ├── composite_cleaner.py
│       │   │   │   ├── jsonl_story_cleaner.py
│       │   │   │   └── steps.py
│       │   │   ├── normalize_text/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── composite_normalizer.py
│       │   │   │   ├── jsonl_story_normalizer.py
│       │   │   │   └── steps.py
│       │   │   ├── tokenize_text/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── jsonl_story_tokenizer.py
│       │   │   │   ├── steps.py
│       │   │   │   └── whitespace_tokenizer.py
│       │   │   └── __init__.py
│       │   ├── writers/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_prediction_writer.py
│       │   └── __init__.py
│       ├── __init__.py
│       └── composition.py
├── tests/
│   ├── acceptance/
│   │   ├── features/
│   │   │   ├── acquiring_a_dataset.feature
│   │   │   ├── cleaning_story_text.feature
│   │   │   ├── normalizing_story_text.feature
│   │   │   ├── preparing_a_dataset.feature
│   │   │   └── preparing_story_files.feature
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── story_text.py
│   │   ├── test_acquiring_a_dataset.py
│   │   ├── test_cleaning_story_text.py
│   │   ├── test_normalizing_story_text.py
│   │   ├── test_preparing_a_dataset.py
│   │   └── test_preparing_story_files.py
│   ├── application/
│   │   ├── use_cases/
│   │   │   ├── __init__.py
│   │   │   └── test_download_dataset.py
│   │   └── __init__.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   └── test_layer_imports.py
│   ├── composition/
│   │   ├── __init__.py
│   │   ├── test_acquire_dataset.py
│   │   ├── test_normalize_dataset.py
│   │   └── test_tokenize_dataset.py
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── test_dataset_spec.py
│   │   │   ├── test_prediction.py
│   │   │   └── test_text.py
│   │   └── __init__.py
│   ├── entrypoints/
│   │   ├── cli/
│   │   │   ├── __init__.py
│   │   │   ├── test_clean_command.py
│   │   │   ├── test_download_command.py
│   │   │   ├── test_normalize_command.py
│   │   │   └── test_tokenize_command.py
│   │   └── __init__.py
│   ├── infrastructure/
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── test_clean_text_config.py
│   │   │   ├── test_dataset_config.py
│   │   │   ├── test_normalize_text_config.py
│   │   │   ├── test_tokenize_config.py
│   │   │   └── test_yaml_config_repository.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── test_filesystem_dataset_repository.py
│   │   │   └── test_huggingface_dataset_source.py
│   │   ├── io/
│   │   │   ├── __init__.py
│   │   │   └── test_read_text_jsonl.py
│   │   ├── readers/
│   │   │   ├── __init__.py
│   │   │   └── test_filesystem_file_reader.py
│   │   ├── runs/
│   │   │   ├── __init__.py
│   │   │   ├── test_download_log_layout.py
│   │   │   └── test_inference_run_layout.py
│   │   ├── text_processing/
│   │   │   ├── clean_text/
│   │   │   │   ├── __init__.py
│   │   │   │   └── test_steps.py
│   │   │   ├── normalize_text/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test_jsonl_story_normalizer.py
│   │   │   │   └── test_steps.py
│   │   │   ├── tokenize_text/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test_jsonl_story_tokenizer.py
│   │   │   │   └── test_steps.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   └── __init__.py
├── .cursorignore
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```
