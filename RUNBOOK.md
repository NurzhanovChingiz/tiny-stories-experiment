# Runbook

## Download TinyStoriesV2

Source: [noanabeshima/TinyStoriesV2](https://huggingface.co/datasets/noanabeshima/TinyStoriesV2)

Published files:

- `TinyStoriesV2-GPT4-train.jsonl` (train split)
- `TinyStoriesV2-GPT4-valid.jsonl` (validation split)

Destination: `data/raw/tiny_stories_raw/`

`data/` is gitignored. The JSONL files are raw inputs. Do not edit them in place. Derived data belongs in `data/interim` or `data/processed`.

```bash
make 01-download
```

The command creates `data/raw/tiny_stories_raw` when it is missing. A file that already has the published byte count is left unchanged. A file that exists with a different size is left unchanged and the command stops. Delete that local file, then run `make 01-download` again.

The train file is about 2.2 GB and the validation file is about 22 MB. Hugging Face Hub cache metadata may appear under `data/raw/tiny_stories_raw/.cache/`. That cache can be deleted; the JSONL files are the raw data.
