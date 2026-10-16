"""Acceptance scenarios for acquiring a published dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx2
import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from typer.testing import CliRunner

from tiny_story.entrypoints.cli.app import app
from tiny_story.entrypoints.cli.download import download

if TYPE_CHECKING:
    from pathlib import Path

scenarios("features/acquiring_a_dataset.feature")

_DATASET_NAME = "other_corpus"
_REPOSITORY = "some-org/other-corpus"
_REVISION = "abc123"
_RUNNER = CliRunner()


@dataclass(frozen=True, slots=True)
class PublishedStory:
    """One story file the publication can provide.

    Attributes:
        name: File name inside the raw collection.
        text: Bytes the publication would copy, as text.
        copy_fails: The copy raises before anything is stored.
        digest_mismatch: The announced digest does not match the copied bytes.
    """

    name: str
    text: str
    copy_fails: bool = False
    digest_mismatch: bool = False


@dataclass(frozen=True, slots=True)
class AcquisitionRequest:
    """Facts the curator supplied before acquiring the dataset.

    Attributes:
        root: Per-scenario directory.
        raw_dir: Raw collection for this dataset.
        overwrite: Replace a raw file that already matches.
        stories: Published story files.
        filenames: File names written into the configuration, when they differ.
        unreachable: The publication cannot be reached.
        configuration_missing: The configuration file is absent.
        prior_bytes: Text already stored in the raw collection.
    """

    root: Path
    raw_dir: Path
    overwrite: bool = False
    stories: tuple[PublishedStory, ...] = ()
    filenames: tuple[str, ...] | None = None
    unreachable: bool = False
    configuration_missing: bool = False
    prior_bytes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """What the curator could observe after one acquisition.

    Attributes:
        raw_dir: Raw collection for this dataset.
        output: Text the command showed the curator.
        refusal: Refusal text. None when acquisition finished.
        exit_code: Command status.
        copied: Story names the publication was asked to copy.
        log_path: Acquisition log for this dataset.
    """

    raw_dir: Path
    output: str
    refusal: str | None
    exit_code: int
    copied: tuple[str, ...]
    log_path: Path


@pytest.fixture(autouse=True)
def _isolate_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep logs and relative paths inside the scenario directory.

    Args:
        tmp_path: Per-test directory.
        monkeypatch: Pytest patch helper used to change the working directory.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fetch_log() -> list[str]:
    """Return the story names copied during this scenario.

    Returns:
        An empty list that the publication stand-in fills.
    """
    return []


@pytest.fixture
def acquisition_request(tmp_path: Path) -> AcquisitionRequest:
    """Return an empty acquisition request in the scenario directory.

    Args:
        tmp_path: Per-test directory.

    Returns:
        A request with no published stories yet.
    """
    return AcquisitionRequest(root=tmp_path, raw_dir=tmp_path / "raw")


def listed_names(request: AcquisitionRequest) -> tuple[str, ...]:
    """Return the file names the configuration will list.

    Args:
        request: Acquisition facts for this scenario.

    Returns:
        Explicit file names, or the published story names.
    """
    if request.filenames is not None:
        return request.filenames
    return tuple(story.name for story in request.stories)


def published_story(request: AcquisitionRequest, name: str) -> PublishedStory:
    """Return one published story by file name.

    Args:
        request: Acquisition facts for this scenario.
        name: File name to find.

    Returns:
        The published story.

    Raises:
        AssertionError: The scenario did not publish that file.
    """
    for story in request.stories:
        if story.name == name:
            return story
    message = f"No published story named {name}"
    raise AssertionError(message)


def sha256_hex(payload: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of ``payload``.

    Args:
        payload: Bytes whose digest the publication announces.

    Returns:
        The lowercase SHA-256 hex digest.
    """
    return hashlib.sha256(payload).hexdigest()


def configuration_path(request: AcquisitionRequest) -> Path:
    """Return the configuration path the curator will open.

    Args:
        request: Acquisition facts for this scenario.

    Returns:
        A missing path when the scenario has no configuration, otherwise the
        scenario configuration file.
    """
    if request.configuration_missing:
        return request.root / "missing.yaml"
    return request.root / "dataset.yaml"


def write_dataset_config(request: AcquisitionRequest, config: Path) -> None:
    """Write the dataset configuration for this scenario.

    Args:
        request: Acquisition facts for this scenario.
        config: Configuration path to write.

    Returns:
        None.
    """
    files = "\n".join(f"    - {name}" for name in listed_names(request))
    flag = "true" if request.overwrite else "false"
    config.write_text(
        f"""
dataset:
  name: {_DATASET_NAME}
  files:
{files}
  destination: {request.raw_dir}
  overwrite: {flag}
  source:
    repository: {_REPOSITORY}
    revision: {_REVISION}
    cache_dir: {request.root / "cache"}
""",
        encoding="utf-8",
    )


def place_prior_files(request: AcquisitionRequest) -> None:
    """Write raw files the curator already has.

    Args:
        request: Acquisition facts for this scenario.

    Returns:
        None.
    """
    if not request.prior_bytes:
        return
    request.raw_dir.mkdir(parents=True, exist_ok=True)
    for name, text in request.prior_bytes:
        (request.raw_dir / name).write_text(text, encoding="utf-8")


def install_publication(
    monkeypatch: pytest.MonkeyPatch,
    request: AcquisitionRequest,
    fetch_log: list[str],
) -> None:
    """Stand in for the publication without contacting the network.

    Args:
        monkeypatch: Pytest patch helper.
        request: Acquisition facts for this scenario.
        fetch_log: Story names the stand-in was asked to copy.

    Returns:
        None.
    """

    def fake_metadata(*_args: object, **_kwargs: object) -> SimpleNamespace:
        if request.unreachable:
            offline = "offline"
            raise httpx2.ConnectError(offline)
        url = _args[0] if _args else ""
        name = str(url).rstrip("/").split("/")[-1]
        story = published_story(request, name)
        announced = f"{story.text}!" if story.digest_mismatch else story.text
        return SimpleNamespace(etag=sha256_hex(announced.encode()))

    def fake_download(**kwargs: object) -> str:
        filename = kwargs["filename"]
        assert isinstance(filename, str)
        fetch_log.append(filename)
        story = published_story(request, filename)
        if story.copy_fails:
            message = "offline"
            raise OSError(message)
        path = request.root / "published" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(story.text.encode())
        return str(path)

    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.get_hf_file_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        "tiny_story.infrastructure.datasets.huggingface_dataset_source.hf_hub_download",
        fake_download,
    )


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


def perform_acquisition(
    request: AcquisitionRequest,
    monkeypatch: pytest.MonkeyPatch,
    fetch_log: list[str],
) -> AcquisitionResult:
    """Acquire the dataset the way the curator would.

    Args:
        request: Acquisition facts for this scenario.
        monkeypatch: Pytest patch helper.
        fetch_log: Story names the publication stand-in copies.

    Returns:
        The observable acquisition result.
    """
    if download.__name__ != "download":
        message = "Dataset download command was not registered"
        raise RuntimeError(message)
    place_prior_files(request)
    config = configuration_path(request)
    if not request.configuration_missing:
        write_dataset_config(request, config)
    install_publication(monkeypatch, request, fetch_log)
    result = _RUNNER.invoke(app, ["download", "--config", str(config)])
    refusal = None if result.exit_code == 0 else exception_text(result)
    return AcquisitionResult(
        raw_dir=request.raw_dir,
        output=result.output,
        refusal=refusal,
        exit_code=result.exit_code,
        copied=tuple(fetch_log),
        log_path=(
            request.root / "artifacts" / "download" / "logs" / f"{_DATASET_NAME}.log"
        ),
    )


def remember_story(
    request: AcquisitionRequest,
    name: str,
    text: str,
    *,
    copy_fails: bool = False,
    digest_mismatch: bool = False,
) -> AcquisitionRequest:
    """Add one published story to the request.

    Args:
        request: Acquisition facts so far.
        name: File name inside the raw collection.
        text: Published story text.
        copy_fails: The copy raises before anything is stored.
        digest_mismatch: The announced digest does not match the copied bytes.

    Returns:
        The request including that story.
    """
    story = PublishedStory(
        name=name,
        text=text,
        copy_fails=copy_fails,
        digest_mismatch=digest_mismatch,
    )
    return replace(request, stories=(*request.stories, story))


@given(
    parsers.parse('a published story file "{name}" that reads "{text}"'),
    target_fixture="acquisition_request",
)
def published_story_that_reads(
    acquisition_request: AcquisitionRequest,
    name: str,
    text: str,
) -> AcquisitionRequest:
    """Publish one story file the curator can acquire.

    Args:
        acquisition_request: Acquisition facts so far.
        name: File name inside the raw collection.
        text: Published story text.

    Returns:
        The request including that story.
    """
    return remember_story(acquisition_request, name, text)


@given(
    parsers.parse('a published story file "{name}" that cannot be copied'),
    target_fixture="acquisition_request",
)
def published_story_that_cannot_be_copied(
    acquisition_request: AcquisitionRequest,
    name: str,
) -> AcquisitionRequest:
    """Publish a story file whose copy fails.

    Args:
        acquisition_request: Acquisition facts so far.
        name: File name inside the raw collection.

    Returns:
        The request including that failing story.
    """
    return remember_story(acquisition_request, name, "notes", copy_fails=True)


@given(
    parsers.parse(
        'a published story file "{name}" whose copy does not match its digest'
    ),
    target_fixture="acquisition_request",
)
def published_story_with_mismatched_digest(
    acquisition_request: AcquisitionRequest,
    name: str,
) -> AcquisitionRequest:
    """Publish a story file whose bytes do not match the announced digest.

    Args:
        acquisition_request: Acquisition facts so far.
        name: File name inside the raw collection.

    Returns:
        The request including that mismatched story.
    """
    return remember_story(acquisition_request, name, "new", digest_mismatch=True)


@given(
    "the curator replaces story files that already match",
    target_fixture="acquisition_request",
)
def curator_replaces_matching_files(
    acquisition_request: AcquisitionRequest,
) -> AcquisitionRequest:
    """Ask acquisition to copy files again even when they match.

    Args:
        acquisition_request: Acquisition facts so far.

    Returns:
        The request with replacement turned on.
    """
    return replace(acquisition_request, overwrite=True)


@given(
    parsers.parse('the raw collection already holds "{name}" reading "{text}"'),
    target_fixture="acquisition_request",
)
def raw_collection_holds(
    acquisition_request: AcquisitionRequest,
    name: str,
    text: str,
) -> AcquisitionRequest:
    """Store one raw file before acquisition.

    Args:
        acquisition_request: Acquisition facts so far.
        name: File name inside the raw collection.
        text: Text already stored for that file.

    Returns:
        The request including that raw file.
    """
    return replace(
        acquisition_request,
        prior_bytes=(*acquisition_request.prior_bytes, (name, text)),
    )


@given(
    parsers.parse('the raw collection already holds a matching "{name}"'),
    target_fixture="acquisition_request",
)
def raw_collection_holds_matching(
    acquisition_request: AcquisitionRequest,
    name: str,
) -> AcquisitionRequest:
    """Store a raw file whose bytes match the publication.

    Args:
        acquisition_request: Acquisition facts so far.
        name: File name inside the raw collection.

    Returns:
        The request including that matching raw file.
    """
    story = published_story(acquisition_request, name)
    return replace(
        acquisition_request,
        prior_bytes=(*acquisition_request.prior_bytes, (name, story.text)),
    )


@given(
    parsers.parse('the raw collection already holds an empty "{name}"'),
    target_fixture="acquisition_request",
)
def raw_collection_holds_empty(
    acquisition_request: AcquisitionRequest,
    name: str,
) -> AcquisitionRequest:
    """Store an empty raw file before acquisition.

    Args:
        acquisition_request: Acquisition facts so far.
        name: File name inside the raw collection.

    Returns:
        The request including that empty raw file.
    """
    return replace(
        acquisition_request,
        prior_bytes=(*acquisition_request.prior_bytes, (name, "")),
    )


@given("no dataset configuration", target_fixture="acquisition_request")
def no_dataset_configuration(
    acquisition_request: AcquisitionRequest,
) -> AcquisitionRequest:
    """Leave the dataset configuration absent.

    Args:
        acquisition_request: Acquisition facts so far.

    Returns:
        The request with no configuration file.
    """
    return replace(acquisition_request, configuration_missing=True)


@given(
    parsers.parse('the dataset lists the file "{name}"'),
    target_fixture="acquisition_request",
)
def dataset_lists_file(
    acquisition_request: AcquisitionRequest,
    name: str,
) -> AcquisitionRequest:
    """List a file name in the dataset configuration.

    Args:
        acquisition_request: Acquisition facts so far.
        name: File name to list.

    Returns:
        The request listing that file.
    """
    return replace(acquisition_request, filenames=(name,))


@given("the publication cannot be reached", target_fixture="acquisition_request")
def publication_cannot_be_reached(
    acquisition_request: AcquisitionRequest,
) -> AcquisitionRequest:
    """Make the publication unreachable for the next acquisition.

    Args:
        acquisition_request: Acquisition facts so far.

    Returns:
        The request whose publication cannot be reached.
    """
    return replace(acquisition_request, unreachable=True)


@given("that story file was recorded from an earlier acquisition")
def story_was_recorded(
    acquisition_request: AcquisitionRequest,
    monkeypatch: pytest.MonkeyPatch,
    fetch_log: list[str],
) -> None:
    """Acquire the published story once so its digest is recorded.

    Args:
        acquisition_request: Acquisition facts so far.
        monkeypatch: Pytest patch helper.
        fetch_log: Story names copied by the earlier acquisition.

    Returns:
        None.
    """
    result = perform_acquisition(acquisition_request, monkeypatch, fetch_log)
    assert result.refusal is None
    fetch_log.clear()


@when("the curator acquires the dataset", target_fixture="acquisition_result")
def curator_acquires_the_dataset(
    acquisition_request: AcquisitionRequest,
    monkeypatch: pytest.MonkeyPatch,
    fetch_log: list[str],
) -> AcquisitionResult:
    """Acquire the configured dataset.

    Args:
        acquisition_request: Acquisition facts for this scenario.
        monkeypatch: Pytest patch helper.
        fetch_log: Story names the publication stand-in copies.

    Returns:
        The observable acquisition result.
    """
    return perform_acquisition(acquisition_request, monkeypatch, fetch_log)


@then(parsers.parse('the curator is told "{name}" was saved'))
def curator_is_told_saved(acquisition_result: AcquisitionResult, name: str) -> None:
    """Assert the command reported one saved raw file.

    Args:
        acquisition_result: Acquisition result for this scenario.
        name: File name the curator was told about.
    """
    assert acquisition_result.refusal is None
    assert acquisition_result.exit_code == 0
    expected = f"saved {acquisition_result.raw_dir / name}"
    assert expected in acquisition_result.output.splitlines()


@then(parsers.parse('the curator is told "{name}" is already present'))
def curator_is_told_present(acquisition_result: AcquisitionResult, name: str) -> None:
    """Assert the command reported one raw file left in place.

    Args:
        acquisition_result: Acquisition result for this scenario.
        name: File name the curator was told about.
    """
    assert acquisition_result.refusal is None
    assert acquisition_result.exit_code == 0
    expected = f"present {acquisition_result.raw_dir / name}"
    assert expected in acquisition_result.output.splitlines()


@then(parsers.parse('the raw file "{name}" reads "{text}"'))
def raw_file_reads(acquisition_result: AcquisitionResult, name: str, text: str) -> None:
    """Assert one raw file's text.

    Args:
        acquisition_result: Acquisition result for this scenario.
        name: File name inside the raw collection.
        text: Text the file must hold.
    """
    assert (acquisition_result.raw_dir / name).read_text(encoding="utf-8") == text


@then(parsers.parse('the raw file "{name}" is absent'))
def raw_file_is_absent(acquisition_result: AcquisitionResult, name: str) -> None:
    """Assert one raw file was not stored.

    Args:
        acquisition_result: Acquisition result for this scenario.
        name: File name inside the raw collection.
    """
    assert not (acquisition_result.raw_dir / name).exists()


@then("the publication was copied")
def publication_was_copied(acquisition_result: AcquisitionResult) -> None:
    """Assert the publication was asked for at least one story file.

    Args:
        acquisition_result: Acquisition result for this scenario.
    """
    assert acquisition_result.copied


@then("the publication was not copied")
def publication_was_not_copied(acquisition_result: AcquisitionResult) -> None:
    """Assert the publication was not asked to copy a story file.

    Args:
        acquisition_result: Acquisition result for this scenario.
    """
    assert acquisition_result.copied == ()


@then(parsers.parse('acquisition is refused because "{reason}"'))
def acquisition_is_refused(acquisition_result: AcquisitionResult, reason: str) -> None:
    """Assert acquisition was refused for the stated reason.

    Args:
        acquisition_result: Acquisition result for this scenario.
        reason: Text the refusal must contain.
    """
    assert acquisition_result.exit_code != 0
    assert acquisition_result.refusal is not None
    assert reason in acquisition_result.refusal


@then("no raw story file is stored")
def no_raw_story_file_is_stored(
    acquisition_request: AcquisitionRequest,
    acquisition_result: AcquisitionResult,
) -> None:
    """Assert none of the configured story files were stored.

    Args:
        acquisition_request: Acquisition facts for this scenario.
        acquisition_result: Acquisition result for this scenario.
    """
    names = listed_names(acquisition_request)
    for name in names:
        stored = acquisition_result.raw_dir / name
        assert not stored.resolve().is_file()
        partial_name = f"{name.replace(chr(92), '/').rsplit('/', 1)[-1]}.partial"
        assert not (acquisition_result.raw_dir / partial_name).exists()
    if not names:
        assert not acquisition_request.raw_dir.exists()


@then(parsers.parse('the acquisition log records "{text}"'))
def acquisition_log_records(acquisition_result: AcquisitionResult, text: str) -> None:
    """Assert the acquisition log contains the stated text.

    Args:
        acquisition_result: Acquisition result for this scenario.
        text: Text the log must contain.
    """
    assert text in acquisition_result.log_path.read_text(encoding="utf-8")
