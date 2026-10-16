"""Acceptance scenarios for preparing a configured dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from typer.testing import CliRunner

from tests.acceptance.story_text import story_body
from tiny_story.entrypoints.cli.app import app
from tiny_story.entrypoints.cli.clean import clean
from tiny_story.entrypoints.cli.normalize import normalize

scenarios("features/preparing_a_dataset.feature")

_DATASET_NAME = "other_corpus"
_STORY_FILE = "valid.jsonl"
_REPOSITORY = "some-org/other-corpus"
_REVISION = "abc123"
_RUNNER = CliRunner()
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLEAN_TEMPLATE = (_REPO_ROOT / "config" / "clean_text.yaml").read_text(
    encoding="utf-8"
)
_NORMALIZE_TEMPLATE = (_REPO_ROOT / "config" / "normalize_text.yaml").read_text(
    encoding="utf-8"
)


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    """Facts the curator supplied before preparing the dataset.

    Attributes:
        root: Per-scenario directory.
        raw_text: Raw JSONL text. None when the scenario has no raw file.
        cleaned_text: Cleaned JSONL text. None when the scenario has no cleaned file.
        raw_missing: The configured raw file is absent.
        cleaned_missing: The configured cleaned file is absent.
        clean_enabled: Story cleaning is on.
        save_clean: Cleaned stories are saved.
        normalize_enabled: Sentence marking is on.
        save_normalize: Marked stories are saved.
        use_default_interim: Write cleaned stories to the interim collection.
        use_default_processed: Read and write the default interim and processed
            collections.
        explicit_clean_log: Record cleaning in a curator-chosen log.
        explicit_normalize_log: Record normalization in a curator-chosen log.
    """

    root: Path
    raw_text: str | None = None
    cleaned_text: str | None = None
    raw_missing: bool = False
    cleaned_missing: bool = False
    clean_enabled: bool = True
    save_clean: bool = True
    normalize_enabled: bool = True
    save_normalize: bool = True
    use_default_interim: bool = False
    use_default_processed: bool = False
    explicit_clean_log: bool = False
    explicit_normalize_log: bool = False


@dataclass(frozen=True, slots=True)
class DatasetResult:
    """What the curator could observe after one preparation command.

    Attributes:
        root: Per-scenario directory.
        output: Text the command showed the curator.
        refusal: Refusal text. None when preparation finished.
        exit_code: Command status.
        raw_file: Configured raw story file.
        raw_original: Raw file text from setup. None when that file was absent.
        interim_file: Cleaned story file.
        processed_file: Marked story file.
        clean_log: Cleaning log for this dataset.
        normalize_log: Normalization log for this dataset.
    """

    root: Path
    output: str
    refusal: str | None
    exit_code: int
    raw_file: Path
    raw_original: str | None
    interim_file: Path
    processed_file: Path
    clean_log: Path
    normalize_log: Path


@pytest.fixture(autouse=True)
def _isolate_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default collections and logs inside the scenario directory.

    Args:
        tmp_path: Per-test directory.
        monkeypatch: Pytest patch helper used to change the working directory.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def dataset_request(tmp_path: Path) -> DatasetRequest:
    """Return an empty preparation request in the scenario directory.

    Args:
        tmp_path: Per-test directory.

    Returns:
        A request with no story files yet.
    """
    return DatasetRequest(root=tmp_path)


def file_text(docstring: str) -> str:
    """Return file text for a Gherkin docstring.

    Args:
        docstring: Docstring attached to a file step.

    Returns:
        The file text with one trailing newline.
    """
    return story_body(docstring) + "\n"


def raw_file(request: DatasetRequest) -> Path:
    """Return the configured raw story path.

    Args:
        request: Preparation facts for this scenario.

    Returns:
        The raw story file.
    """
    return request.root / "raw" / _STORY_FILE


def interim_file(request: DatasetRequest) -> Path:
    """Return the cleaned story path for this scenario.

    Args:
        request: Preparation facts for this scenario.

    Returns:
        The interim story file.
    """
    if request.use_default_interim or request.use_default_processed:
        return request.root / "data" / "interim" / _DATASET_NAME / _STORY_FILE
    return request.root / "interim" / _STORY_FILE


def processed_file(request: DatasetRequest) -> Path:
    """Return the marked story path for this scenario.

    Args:
        request: Preparation facts for this scenario.

    Returns:
        The processed story file.
    """
    if request.use_default_processed:
        return request.root / "data" / "processed" / _DATASET_NAME / _STORY_FILE
    return request.root / "processed" / _STORY_FILE


def clean_log(request: DatasetRequest) -> Path:
    """Return the cleaning log path for this scenario.

    Args:
        request: Preparation facts for this scenario.

    Returns:
        The explicit log, or the default cleaning log.
    """
    if request.explicit_clean_log:
        return request.root / "custom" / "clean.log"
    return request.root / "artifacts" / "clean" / "logs" / f"{_DATASET_NAME}.log"


def normalize_log(request: DatasetRequest) -> Path:
    """Return the normalization log path for this scenario.

    Args:
        request: Preparation facts for this scenario.

    Returns:
        The explicit log, or the default normalization log.
    """
    if request.explicit_normalize_log:
        return request.root / "custom" / "normalize.log"
    return request.root / "artifacts" / "normalize" / "logs" / f"{_DATASET_NAME}.log"


def render_switches(template: str, *, enabled: bool, save_processed: bool) -> str:
    """Return configuration text with the scenario switches applied.

    Args:
        template: Shipped configuration text.
        enabled: Whether preparation runs.
        save_processed: Whether prepared stories are saved.

    Returns:
        The configuration text.
    """
    enabled_text = "true" if enabled else "false"
    save_text = "true" if save_processed else "false"
    rendered = template.replace("enabled: true", f"enabled: {enabled_text}", 1)
    return rendered.replace("save_processed: true", f"save_processed: {save_text}", 1)


def write_dataset_config(request: DatasetRequest, config: Path) -> None:
    """Write the dataset configuration for this scenario.

    Args:
        request: Preparation facts for this scenario.
        config: Configuration path to write.

    Returns:
        None.
    """
    config.write_text(
        f"""
dataset:
  name: {_DATASET_NAME}
  files:
    - {_STORY_FILE}
  destination: {request.root / "raw"}
  overwrite: false
  source:
    repository: {_REPOSITORY}
    revision: {_REVISION}
    cache_dir: {request.root / "cache"}
""",
        encoding="utf-8",
    )


def write_text_file(path: Path, text: str) -> None:
    """Write one story file, creating its directory.

    Args:
        path: File to write.
        text: File text.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def place_story_files(request: DatasetRequest) -> str | None:
    """Write the raw and cleaned files the scenario already has.

    Args:
        request: Preparation facts for this scenario.

    Returns:
        The raw file text, or None when that file was left absent.
    """
    original = None
    if request.raw_text is not None and not request.raw_missing:
        original = request.raw_text
        write_text_file(raw_file(request), original)
    if request.cleaned_text is not None and not request.cleaned_missing:
        write_text_file(interim_file(request), request.cleaned_text)
    return original


def exception_text(result: object) -> str:
    """Return the refusal text from a failed command.

    Args:
        result: Typer runner result.

    Returns:
        The exception text, or the command output when no exception was kept.
    """
    exception = getattr(result, "exception", None)
    if exception is None:
        output = getattr(result, "output", "")
        return str(output)
    return str(exception)


def command_result(
    request: DatasetRequest,
    arguments: list[str],
    original: str | None,
) -> DatasetResult:
    """Run one dataset command and keep the observable result.

    Args:
        request: Preparation facts for this scenario.
        arguments: Command arguments.
        original: Raw file text from setup.

    Returns:
        The observable preparation result.
    """
    result = _RUNNER.invoke(app, arguments)
    refusal = None if result.exit_code == 0 else exception_text(result)
    return DatasetResult(
        root=request.root,
        output=result.output,
        refusal=refusal,
        exit_code=result.exit_code,
        raw_file=raw_file(request),
        raw_original=original,
        interim_file=interim_file(request),
        processed_file=processed_file(request),
        clean_log=clean_log(request),
        normalize_log=normalize_log(request),
    )


def clean_dataset_command(request: DatasetRequest) -> DatasetResult:
    """Clean the configured dataset the way the curator would.

    Args:
        request: Preparation facts for this scenario.

    Returns:
        The observable cleaning result.
    """
    if clean.__name__ != "clean":
        message = "Dataset clean command was not registered"
        raise RuntimeError(message)
    original = place_story_files(request)
    dataset_config = request.root / "dataset.yaml"
    clean_config = request.root / "clean_text.yaml"
    write_dataset_config(request, dataset_config)
    clean_config.write_text(
        render_switches(
            _CLEAN_TEMPLATE,
            enabled=request.clean_enabled,
            save_processed=request.save_clean,
        ),
        encoding="utf-8",
    )
    arguments = [
        "clean",
        "--dataset-config",
        str(dataset_config),
        "--config",
        str(clean_config),
    ]
    if not request.use_default_interim:
        arguments.extend(["--output-dir", str(request.root / "interim")])
    if request.explicit_clean_log:
        arguments.extend(["--log-path", str(clean_log(request))])
    return command_result(request, arguments, original)


def normalize_dataset_command(request: DatasetRequest) -> DatasetResult:
    """Normalize the configured dataset the way the curator would.

    Args:
        request: Preparation facts for this scenario.

    Returns:
        The observable normalization result.
    """
    if normalize.__name__ != "normalize":
        message = "Dataset normalize command was not registered"
        raise RuntimeError(message)
    original = place_story_files(request)
    dataset_config = request.root / "dataset.yaml"
    normalize_config = request.root / "normalize_text.yaml"
    write_dataset_config(request, dataset_config)
    normalize_config.write_text(
        render_switches(
            _NORMALIZE_TEMPLATE,
            enabled=request.normalize_enabled,
            save_processed=request.save_normalize,
        ),
        encoding="utf-8",
    )
    arguments = [
        "normalize",
        "--dataset-config",
        str(dataset_config),
        "--config",
        str(normalize_config),
    ]
    if not request.use_default_processed:
        arguments.extend(
            [
                "--input-dir",
                str(request.root / "interim"),
                "--output-dir",
                str(request.root / "processed"),
            ]
        )
    if request.explicit_normalize_log:
        arguments.extend(["--log-path", str(normalize_log(request))])
    return command_result(request, arguments, original)


def story_text(path: Path) -> str:
    """Return the text field of one JSONL story file.

    Args:
        path: JSONL file with one story object.

    Returns:
        The story text.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = payload["text"]
    assert isinstance(text, str)
    return text


@given("a configured raw story file:", target_fixture="dataset_request")
def configured_raw_story_file(
    dataset_request: DatasetRequest,
    docstring: str,
) -> DatasetRequest:
    """Store the raw story file named by the dataset configuration.

    Args:
        dataset_request: Preparation facts so far.
        docstring: JSONL contents from the scenario.

    Returns:
        The request including that raw file.
    """
    return replace(dataset_request, raw_text=file_text(docstring), raw_missing=False)


@given("a configured cleaned story file:", target_fixture="dataset_request")
def configured_cleaned_story_file(
    dataset_request: DatasetRequest,
    docstring: str,
) -> DatasetRequest:
    """Store the cleaned story file named by the dataset configuration.

    Args:
        dataset_request: Preparation facts so far.
        docstring: JSONL contents from the scenario.

    Returns:
        The request including that cleaned file.
    """
    return replace(
        dataset_request,
        cleaned_text=file_text(docstring),
        cleaned_missing=False,
    )


@given(
    "a configured cleaned story file in the interim collection:",
    target_fixture="dataset_request",
)
def configured_cleaned_story_in_interim(
    dataset_request: DatasetRequest,
    docstring: str,
) -> DatasetRequest:
    """Store the cleaned story in the default interim collection.

    Args:
        dataset_request: Preparation facts so far.
        docstring: JSONL contents from the scenario.

    Returns:
        The request including that interim file.
    """
    return replace(
        dataset_request,
        cleaned_text=file_text(docstring),
        cleaned_missing=False,
        use_default_processed=True,
    )


@given("saving cleaned stories is turned off", target_fixture="dataset_request")
def saving_cleaned_stories_is_off(dataset_request: DatasetRequest) -> DatasetRequest:
    """Turn off saving cleaned stories.

    Args:
        dataset_request: Preparation facts so far.

    Returns:
        The request with cleaned stories unsaved.
    """
    return replace(dataset_request, save_clean=False)


@given("saving marked stories is turned off", target_fixture="dataset_request")
def saving_marked_stories_is_off(dataset_request: DatasetRequest) -> DatasetRequest:
    """Turn off saving marked stories.

    Args:
        dataset_request: Preparation facts so far.

    Returns:
        The request with marked stories unsaved.
    """
    return replace(dataset_request, save_normalize=False)


@given(
    "saving cleaned stories is turned on while cleaning is off",
    target_fixture="dataset_request",
)
def saving_cleaned_while_cleaning_is_off(
    dataset_request: DatasetRequest,
) -> DatasetRequest:
    """Ask to save cleaned stories while cleaning is off.

    Args:
        dataset_request: Preparation facts so far.

    Returns:
        The request with that inconsistent switch.
    """
    return replace(dataset_request, clean_enabled=False, save_clean=True)


@given(
    "saving marked stories is turned on while sentence marking is off",
    target_fixture="dataset_request",
)
def saving_marked_while_marking_is_off(
    dataset_request: DatasetRequest,
) -> DatasetRequest:
    """Ask to save marked stories while sentence marking is off.

    Args:
        dataset_request: Preparation facts so far.

    Returns:
        The request with that inconsistent switch.
    """
    return replace(dataset_request, normalize_enabled=False, save_normalize=True)


@given("the configured raw story file is missing", target_fixture="dataset_request")
def configured_raw_story_is_missing(dataset_request: DatasetRequest) -> DatasetRequest:
    """Leave the configured raw story file absent.

    Args:
        dataset_request: Preparation facts so far.

    Returns:
        The request with no raw story file.
    """
    return replace(dataset_request, raw_missing=True, raw_text=None)


@given("the configured cleaned story file is missing", target_fixture="dataset_request")
def configured_cleaned_story_is_missing(
    dataset_request: DatasetRequest,
) -> DatasetRequest:
    """Leave the configured cleaned story file absent.

    Args:
        dataset_request: Preparation facts so far.

    Returns:
        The request with no cleaned story file.
    """
    return replace(dataset_request, cleaned_missing=True, cleaned_text=None)


@given("an explicit cleaning log", target_fixture="dataset_request")
def explicit_cleaning_log(dataset_request: DatasetRequest) -> DatasetRequest:
    """Record cleaning in a curator-chosen log.

    Args:
        dataset_request: Preparation facts so far.

    Returns:
        The request with an explicit cleaning log.
    """
    return replace(dataset_request, explicit_clean_log=True)


@given("an explicit normalization log", target_fixture="dataset_request")
def explicit_normalization_log(dataset_request: DatasetRequest) -> DatasetRequest:
    """Record normalization in a curator-chosen log.

    Args:
        dataset_request: Preparation facts so far.

    Returns:
        The request with an explicit normalization log.
    """
    return replace(dataset_request, explicit_normalize_log=True)


@when("the curator cleans the dataset", target_fixture="dataset_result")
def curator_cleans_the_dataset(dataset_request: DatasetRequest) -> DatasetResult:
    """Clean the configured dataset into the scenario directory.

    Args:
        dataset_request: Preparation facts for this scenario.

    Returns:
        The observable cleaning result.
    """
    return clean_dataset_command(dataset_request)


@when(
    "the curator cleans the dataset into the interim collection",
    target_fixture="dataset_result",
)
def curator_cleans_into_interim(dataset_request: DatasetRequest) -> DatasetResult:
    """Clean the configured dataset into the default interim collection.

    Args:
        dataset_request: Preparation facts for this scenario.

    Returns:
        The observable cleaning result.
    """
    return clean_dataset_command(replace(dataset_request, use_default_interim=True))


@when("the curator normalizes the dataset", target_fixture="dataset_result")
def curator_normalizes_the_dataset(dataset_request: DatasetRequest) -> DatasetResult:
    """Normalize the configured dataset into the scenario directory.

    Args:
        dataset_request: Preparation facts for this scenario.

    Returns:
        The observable normalization result.
    """
    return normalize_dataset_command(dataset_request)


@when(
    "the curator normalizes the dataset into the processed collection",
    target_fixture="dataset_result",
)
def curator_normalizes_into_processed(
    dataset_request: DatasetRequest,
) -> DatasetResult:
    """Normalize the configured dataset into the default processed collection.

    Args:
        dataset_request: Preparation facts for this scenario.

    Returns:
        The observable normalization result.
    """
    return normalize_dataset_command(
        replace(dataset_request, use_default_processed=True)
    )


@then(parsers.parse('the interim story reads "{text}"'))
def interim_story_reads(dataset_result: DatasetResult, text: str) -> None:
    """Assert the cleaned story text.

    Args:
        dataset_result: Preparation result for this scenario.
        text: Text the cleaned story must hold.
    """
    assert dataset_result.refusal is None
    assert story_text(dataset_result.interim_file) == text


@then(parsers.parse('the processed story reads "{text}"'))
def processed_story_reads(dataset_result: DatasetResult, text: str) -> None:
    """Assert the marked story text.

    Args:
        dataset_result: Preparation result for this scenario.
        text: Text the marked story must hold.
    """
    assert dataset_result.refusal is None
    assert story_text(dataset_result.processed_file) == text


@then("the configured raw story file is unchanged")
def configured_raw_story_is_unchanged(dataset_result: DatasetResult) -> None:
    """Assert the raw file still holds its original text.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    assert dataset_result.raw_original is not None
    assert (
        dataset_result.raw_file.read_text(encoding="utf-8")
        == dataset_result.raw_original
    )


@then("the curator is told the interim file was saved")
def curator_is_told_interim_saved(dataset_result: DatasetResult) -> None:
    """Assert the command reported the cleaned file.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    assert dataset_result.exit_code == 0
    expected = f"saved {dataset_result.interim_file}"
    assert expected in dataset_result.output.splitlines()


@then("the curator is told the processed file was saved")
def curator_is_told_processed_saved(dataset_result: DatasetResult) -> None:
    """Assert the command reported the marked file.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    assert dataset_result.exit_code == 0
    expected = f"saved {dataset_result.processed_file}"
    assert expected in dataset_result.output.splitlines()


@then("the interim story is stored in the interim collection")
def interim_story_is_in_the_collection(dataset_result: DatasetResult) -> None:
    """Assert the cleaned story is in the default interim collection.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    expected = dataset_result.root / "data" / "interim" / _DATASET_NAME / _STORY_FILE
    assert dataset_result.interim_file == expected
    assert dataset_result.interim_file.is_file()


@then("the processed story is stored in the processed collection")
def processed_story_is_in_the_collection(dataset_result: DatasetResult) -> None:
    """Assert the marked story is in the default processed collection.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    expected = dataset_result.root / "data" / "processed" / _DATASET_NAME / _STORY_FILE
    assert dataset_result.processed_file == expected
    assert dataset_result.processed_file.is_file()


@then("no interim story file is written")
def no_interim_story_file_is_written(dataset_result: DatasetResult) -> None:
    """Assert cleaning did not create the interim story.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    assert dataset_result.exit_code == 0
    assert not dataset_result.interim_file.exists()
    assert not dataset_result.interim_file.parent.exists()


@then("no processed story file is written")
def no_processed_story_file_is_written(dataset_result: DatasetResult) -> None:
    """Assert normalization did not create the processed story.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    assert dataset_result.exit_code == 0
    assert not dataset_result.processed_file.exists()
    assert not dataset_result.processed_file.parent.exists()


@then(parsers.parse('dataset preparation is refused because "{reason}"'))
def dataset_preparation_is_refused(dataset_result: DatasetResult, reason: str) -> None:
    """Assert preparation was refused for the stated reason.

    Args:
        dataset_result: Preparation result for this scenario.
        reason: Text the refusal must contain.
    """
    assert dataset_result.exit_code != 0
    assert dataset_result.refusal is not None
    assert reason in dataset_result.refusal


@then(parsers.parse('the cleaning log records "{text}"'))
def cleaning_log_records(dataset_result: DatasetResult, text: str) -> None:
    """Assert the cleaning log contains the stated text.

    Args:
        dataset_result: Preparation result for this scenario.
        text: Text the log must contain.
    """
    assert text in dataset_result.clean_log.read_text(encoding="utf-8")


@then(parsers.parse('the normalization log records "{text}"'))
def normalization_log_records(dataset_result: DatasetResult, text: str) -> None:
    """Assert the normalization log contains the stated text.

    Args:
        dataset_result: Preparation result for this scenario.
        text: Text the log must contain.
    """
    assert text in dataset_result.normalize_log.read_text(encoding="utf-8")


@then(parsers.parse('the explicit cleaning log records "{text}"'))
def explicit_cleaning_log_records(dataset_result: DatasetResult, text: str) -> None:
    """Assert the curator-chosen cleaning log contains the stated text.

    Args:
        dataset_result: Preparation result for this scenario.
        text: Text the log must contain.
    """
    assert dataset_result.clean_log == dataset_result.root / "custom" / "clean.log"
    assert text in dataset_result.clean_log.read_text(encoding="utf-8")


@then("the default cleaning log is not created")
def default_cleaning_log_is_not_created(dataset_result: DatasetResult) -> None:
    """Assert cleaning did not create the default log directory.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    assert not (dataset_result.root / "artifacts").exists()


@then(parsers.parse('the explicit normalization log records "{text}"'))
def explicit_normalization_log_records(
    dataset_result: DatasetResult,
    text: str,
) -> None:
    """Assert the curator-chosen normalization log contains the stated text.

    Args:
        dataset_result: Preparation result for this scenario.
        text: Text the log must contain.
    """
    expected = dataset_result.root / "custom" / "normalize.log"
    assert dataset_result.normalize_log == expected
    assert text in dataset_result.normalize_log.read_text(encoding="utf-8")


@then("the default normalization log is not created")
def default_normalization_log_is_not_created(dataset_result: DatasetResult) -> None:
    """Assert normalization did not create the default log directory.

    Args:
        dataset_result: Preparation result for this scenario.
    """
    assert not (dataset_result.root / "artifacts").exists()
