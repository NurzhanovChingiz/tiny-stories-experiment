# Tiny Stories experiment

This repository acquires raw datasets for later experiments. Downloading is separate from preprocessing, tokenization, and training.

## Architecture

Acquisition follows the repository layering:

`entrypoints → composition → application → domain`

`infrastructure → application/domain`

| Responsibility | Where it lives |
| --- | --- |
| Dataset identity (`name`, raw file names) | `domain/dataset/dataset_spec.py` |
| Acquisition use case and existing-file policy | `application/use_cases/download_dataset.py` |
| Source port (`DatasetSource.fetch`) | `application/ports/dataset_source.py` |
| Raw storage port and filesystem store | `application/ports/raw_dataset_store.py`, `infrastructure/datasets/filesystem_dataset_repository.py` |
| Hugging Face download | `infrastructure/datasets/huggingface_dataset_source.py` |
| YAML parsing | `infrastructure/config/dataset_config.py` |
| Wiring | `composition.py` |

The use case does not import Hugging Face. It validates the dataset spec, skips or replaces complete raw files, asks the injected source for each pending file, and stores the bytes. `composition.py` reads the configuration and constructs `HuggingFaceDatasetSource`.

## Hugging Face

`dataset.source` names one Hugging Face dataset repository. Downloads happen one file at a time. `revision` is passed through to the Hub. The default revision is `main`; pin a commit hash when a run must be reproducible. `cache_dir` is only an interim Hub cache. The default cache directory is `data/interim/huggingface`.

```yaml
dataset:
  name: tiny_stories
  files:
    - TinyStoriesV2-GPT4-train.jsonl
    - TinyStoriesV2-GPT4-valid.jsonl
  destination: data/raw/tiny_stories_raw
  overwrite: false
  source:
    repository: noanabeshima/TinyStoriesV2
    revision: main
    cache_dir: data/interim/huggingface
```

Another Hub dataset uses the same adapter. Change the configuration; do not add a dataset class:

```yaml
dataset:
  name: other_corpus
  files:
    - notes.txt
  destination: data/raw/other_corpus
  source:
    repository: some-org/other-corpus
    revision: abc123
```

There is no separate split field. Choose the published file names that correspond to the split you want.

## Raw storage

Each acquired file is stored at `{dataset.destination}/{filename}`.

- Acquired bytes are copied to a unique temporary file in the destination directory and published with an atomic rename. A failed copy removes that temporary file and leaves an existing final file unchanged.
- A non-empty regular file is treated as already present. Its checksum and completeness against the remote dataset are not verified. An empty file is acquired again.
- A symbolic link at the destination name is rejected, so a link to a file outside the raw directory is not treated as a finished download.
- A file whose name ends in `.partial` is a normal dataset file. Storing another file does not delete it. There is no resume of a partial body.
- `overwrite: false` (the default) leaves complete files in place and acquires only missing ones.
- `overwrite: true` acquires every listed file again.
- A failure while acquiring one file does not report success. Files that were already published in that run stay in place; the failed file is retried on the next run.

## Command

`make download` reads `config/dataset.yaml` and downloads TinyStoriesV2.

```bash
uv run python -m tiny_story.entrypoints.cli download
uv run python -m tiny_story.entrypoints.cli download --config config/dataset.yaml
uv run python -m tiny_story.entrypoints.cli download --raw-dir data/raw/tiny_stories_raw --cache-dir data/interim/huggingface
```

`--raw-dir` replaces `dataset.destination`. `--cache-dir` replaces `dataset.source.cache_dir`.

## Compatibility

This acquisition path no longer hard-codes TinyStories inside the domain or the use case.

- `DatasetSpec` now has `name` and `filenames`. It no longer has `repo_id` or `source_url`.
- `PublishedDatasetSource.fetch(repo_id, filename)` is now `DatasetSource.fetch(filename)`. The repository and revision belong to the Hugging Face adapter.
- `HuggingfaceTinystoriesRepository` is now `HuggingFaceDatasetSource`, which accepts any Hub dataset id.
- `download_tinystories_v2(raw_dir, cache_dir)` is now `acquire_dataset(config_path, destination=..., cache_dir=...)`.
- `make download` still downloads `noanabeshima/TinyStoriesV2` because that is what `config/dataset.yaml` selects.
- `--raw-dir` and `--cache-dir` remain. `--config` selects the dataset file. The default config path is `config/dataset.yaml`.
