# Runbook

## Download TinyStoriesV2

Source: [noanabeshima/TinyStoriesV2](https://huggingface.co/datasets/noanabeshima/TinyStoriesV2)

Published files:

- `TinyStoriesV2-GPT4-train.jsonl` (train split)
- `TinyStoriesV2-GPT4-valid.jsonl` (validation split)

Destination: `data/raw/tiny_stories_raw/`

`data/` is gitignored. The JSONL files are raw inputs. Do not edit them in place. Derived data belongs in `data/interim` or `data/processed`.

```bash
make download
```

The command creates `data/raw/tiny_stories_raw` when it is missing. A file that already has the published byte count is left unchanged. A file that exists with a different size is left unchanged and the command stops. Delete that local file, then run `make download` again.

The train file is about 2.2 GB and the validation file is about 22 MB. Hugging Face Hub cache metadata may appear under `data/raw/tiny_stories_raw/.cache/`. That cache can be deleted; the JSONL files are the raw data.

## Prepare derived split files

Reads the raw JSONL files and writes derived story text. Raw files stay at the same byte size. Stories that contain line breaks stay one JSON record.

Destination:

- `data/processed/tiny_stories/train.jsonl`
- `data/processed/tiny_stories/valid.jsonl`

```bash
make prepare
```

The command creates `data/processed/tiny_stories` when it is missing. Each derived file is replaced by a completed rewrite. A missing raw file, a raw file whose name does not end in `train` or `valid`, or a line that is not a JSON object with a string `text` field stops the command.
