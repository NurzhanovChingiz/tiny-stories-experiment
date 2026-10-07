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

The command creates `data/raw/tiny_stories_raw` when it is missing. A file that already has the published byte count is left unchanged. A file that already exists with a different size is left unchanged and the command stops. Delete that local file, then run `make download` again. A download that finishes at a different size is removed so the next run can fetch it.

The train file is about 2.2 GB and the validation file is about 22 MB. Hugging Face Hub cache metadata may appear under `data/raw/tiny_stories_raw/.cache/`. That cache can be deleted; the JSONL files are the raw data.

## Prepare derived split files

Reads the raw JSONL files and writes derived story text. Raw files stay at the same byte size. Stories that contain line breaks stay one JSON record.

Destination:

- `data/processed/tiny_stories/train.jsonl`
- `data/processed/tiny_stories/valid.jsonl`

```bash
make prepare
```

The command creates `data/processed/tiny_stories` when it is missing. Derived files are replaced only after every split has been staged and the raw byte counts still match. A missing raw file, a raw file whose name does not end in `train` or `valid`, a line that is not a JSON object with a string `text` field, or a raw size change stops the command and leaves the previous derived files in place. A story-line failure stays the raised error when a raw size change happens during that failure.

## Train a tokenizer

Reads `data/processed/tiny_stories/train.jsonl` and `valid.jsonl`. Writes a byte-level BPE tokenizer. Those prepared files stay at the same byte size. The command does not open the raw files.

Destination: `artifacts/tokenizers/tiny_stories_bpe/tokenizer.json`

```bash
make train-tokenizer
```

The default vocabulary size is 4096, including the reserved token `<|endoftext|>`. A pair must appear at least twice to become a merge. The command creates the artifact directory when it is missing. It replaces `tokenizer.json` only after training finishes and both prepared files still have the byte sizes recorded at the start. A missing split, a line that is not a JSON object with a string `text` field, an empty corpus, a vocabulary size below 257, or a prepared size change stops the command and leaves the previous `tokenizer.json` in place.
