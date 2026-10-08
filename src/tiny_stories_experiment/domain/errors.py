"""Domain errors for dataset download, preparation, tokenizer training, and model specs."""


class RawFileConflictError(Exception):
    """Local raw file size differs from the published file."""


class DownloadSizeMismatchError(Exception):
    """Fetched file size differs from the published byte count."""


class PublishedSizeMissingError(Exception):
    """Hub metadata for a published file did not include a byte count."""


class RawDatasetMissingError(Exception):
    """A raw file named by the dataset spec is absent."""


class UnknownDatasetSplitError(Exception):
    """A raw file name does not end in a published split token."""


class RawDatasetChangedError(Exception):
    """A raw file's size changed while derived text was written."""


class RawStoryRecordError(Exception):
    """A raw JSONL line is not an object with a string text field."""


class InvalidTokenizerSpecError(Exception):
    """A tokenizer spec cannot be trained or stored."""


class InvalidModelSpecError(Exception):
    """A causal language model spec cannot be built."""


class InvalidTokenBatchError(Exception):
    """A token-id batch cannot be scored by a model spec."""


class ProcessedTextMissingError(Exception):
    """A prepared split file is absent."""


class ProcessedTextChangedError(Exception):
    """A prepared split file's size changed while the tokenizer was trained."""


class TokenizerTrainingError(Exception):
    """BPE training could not produce a tokenizer."""
