"""Filesystem adapter for unmodified raw dataset files."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from tiny_story.application.ports.raw_dataset_store import RawDatasetStore
from tiny_story.domain.entities.dataset_spec import require_raw_filename
from tiny_story.domain.errors import RawStorageError

if TYPE_CHECKING:
    from collections.abc import Iterator

_HOLDS_DIRECTORY_NAME = ".holds"
_HOLD_FILES_DIRECTORY_NAME = "files"
_PUBLISH_LOCK_NAME = "lock"
_DIGESTS_DIRECTORY_NAME = ".digests"
_SHA256_HEX_LENGTH = 64


class FilesystemDatasetRepository(RawDatasetStore):
    """Copies acquired files into a raw directory."""

    def __init__(self, raw_dir: Path) -> None:
        """Store the raw directory.

        Args:
            raw_dir: Directory that receives unmodified dataset files.

        Returns:
            None.
        """
        self._raw_dir = raw_dir
        self._publish_lock_depth = 0

    @property
    def directory(self) -> Path:
        """Directory that receives unmodified raw files.

        Returns:
            The raw directory.
        """
        return self._raw_dir

    @contextmanager
    def hold_publication(self) -> Iterator[None]:
        """Hold exclusive publication across every file in one acquisition.

        Yields:
            None while publication is held exclusively.
        """
        with self._acquire_publish_lock():
            yield

    def existing_file(
        self,
        filename: str,
        content_sha256: str,
        publication_id: str,
    ) -> Path | None:
        """Return the stored file when its bytes match a published digest.

        Args:
            filename: Raw file name.
            content_sha256: Lowercase SHA-256 hex digest of the published bytes.
            publication_id: Publication whose digest is being recorded.

        Returns:
            The raw path, or None when that file is absent or has a different digest.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link, or the digest
                record could not be read or written.

        Flow:
            1. Hold exclusive publication.
            2. Return the path when the stored bytes match the digest.
        """
        # 1. Hold exclusive publication.
        with self._acquire_publish_lock():
            # 2. Return the path when the stored bytes match the digest.
            return self._matching_file(filename, content_sha256, publication_id)

    def verified_file(self, filename: str, publication_id: str) -> Path | None:
        """Return the stored file when it still matches one publication's record.

        Args:
            filename: Raw file name.
            publication_id: Publication the recorded digest must name.

        Returns:
            The raw path, or None when that file is absent, has no record for
            ``publication_id``, or no longer matches the recorded digest.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link, or the digest
                record could not be read.

        Flow:
            1. Hold exclusive publication.
            2. Return the path when the stored bytes match the recorded digest.
        """
        # 1. Hold exclusive publication.
        with self._acquire_publish_lock():
            # 2. Return the path when the stored bytes match the recorded digest.
            return self._verified_publication(filename, publication_id)

    def store_unchanged(
        self,
        filename: str,
        source_file: Path,
        content_sha256: str,
        publication_id: str,
    ) -> Path:
        """Copy acquired bytes into the raw directory when they match a digest.

        Args:
            filename: Raw file name.
            source_file: Local file whose bytes are copied.
            content_sha256: Lowercase SHA-256 hex digest the copy must match.
            publication_id: Publication recorded with the verified digest.

        Returns:
            The raw path that now holds those bytes.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link, the bytes could
                not be copied, or the digest record could not be written.

        Flow:
            1. Hold exclusive publication.
            2. Copy verified bytes and record the digest.
        """
        # 1. Hold exclusive publication.
        with self._acquire_publish_lock():
            # 2. Copy verified bytes and record the digest.
            return self._publish_copy(
                filename,
                source_file,
                content_sha256,
                publication_id,
            )

    def _matching_file(
        self,
        filename: str,
        content_sha256: str,
        publication_id: str,
    ) -> Path | None:
        """Return a stored file when its bytes match a digest.

        Args:
            filename: Raw file name.
            content_sha256: Lowercase SHA-256 hex digest of the published bytes.
            publication_id: Publication whose digest is being recorded.

        Returns:
            The raw path, or None when that file is absent or has a different digest.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link, or the digest
                record could not be read or written.

        Flow:
            1. Reject an unsafe filename or a destination symbolic link.
            2. Leave a missing file's record alone.
            3. Clear the record when the stored bytes do not match.
            4. Record the digest and return the matching path.
        """
        # 1. Reject an unsafe filename or a destination symbolic link.
        path = self._regular_destination(filename)
        self._reject_locked_targets(filename)

        # 2. Leave a missing file's record alone.
        if not path.is_file():
            return None

        # 3. Clear the record when the stored bytes do not match.
        if not _digest_matches(_sha256_hex(path), content_sha256):
            self._clear_publication_record(filename)
            return None

        # 4. Record the digest and return the matching path.
        self._publish_digest(filename, content_sha256, publication_id)
        return path

    def _verified_publication(self, filename: str, publication_id: str) -> Path | None:
        """Return a stored file that still matches one publication's record.

        Args:
            filename: Raw file name.
            publication_id: Publication the recorded digest must name.

        Returns:
            The raw path, or None when that file is absent, has no record for
            ``publication_id``, or no longer matches the recorded digest.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link, or the digest
                record could not be read.

        Flow:
            1. Reject an unsafe filename or a destination symbolic link.
            2. Recover a held copy when the raw file is absent.
            3. Return the path when the stored bytes match the recorded digest.
        """
        # 1. Reject an unsafe filename or a destination symbolic link.
        path = self._regular_destination(filename)
        self._reject_locked_targets(filename)

        # 2. Recover a held copy when the raw file is absent.
        if not path.is_file():
            self._restore_held_file(filename)
        if not path.is_file():
            return None

        # 3. Return the path when the stored bytes match the recorded digest.
        record = self._read_digest_record(filename)
        if record is None or record[0] != publication_id:
            return None
        if not _digest_matches(_sha256_hex(path), record[1]):
            return None
        return path

    def _publish_copy(
        self,
        filename: str,
        source_file: Path,
        content_sha256: str,
        publication_id: str,
    ) -> Path:
        """Copy source bytes into the raw directory when they match a digest.

        Args:
            filename: Raw file name.
            source_file: Local file whose bytes are copied.
            content_sha256: Lowercase SHA-256 hex digest the copy must match.
            publication_id: Publication recorded with the verified digest.

        Returns:
            The raw path that now holds those bytes.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link, the bytes could
                not be copied, or the digest record could not be written.

        Flow:
            1. Reject an unsafe filename or a destination symbolic link.
            2. Hold the previous regular file while the new bytes are copied.
            3. Commit the verified copy and record the digest.
        """
        # 1. Reject an unsafe filename or a destination symbolic link.
        destination = self._regular_destination(filename)
        self._reject_locked_targets(filename)

        # 2. Hold the previous regular file while the new bytes are copied.
        held = self._hold_previous_bytes(filename, destination)
        try:
            temporary = self._copy_to_temporary(filename, source_file)
            self._require_matching_digest(filename, temporary, content_sha256)
        except RawStorageError:
            if held is not None:
                held.unlink(missing_ok=True)
            raise

        # 3. Commit the verified copy and record the digest.
        return self._commit_verified_copy(
            filename,
            destination,
            temporary,
            held=held,
            content_sha256=content_sha256,
            publication_id=publication_id,
        )

    def _commit_verified_copy(
        self,
        filename: str,
        destination: Path,
        temporary: Path,
        *,
        held: Path | None,
        content_sha256: str,
        publication_id: str,
    ) -> Path:
        """Publish a verified temporary file and record its digest.

        Args:
            filename: Raw file name.
            destination: Raw path that receives the copy.
            temporary: Verified temporary file.
            held: Previous bytes, or None when there was nothing to hold.
            content_sha256: Lowercase SHA-256 hex digest to record.
            publication_id: Publication recorded with the digest.

        Returns:
            The raw path that now holds those bytes.

        Raises:
            RawStorageError: The bytes or digest record could not be published.

        Flow:
            1. Recheck the destination and publish the temporary file.
            2. Record the digest, then remove the held previous bytes.
        """
        # 1. Recheck the destination and publish the temporary file.
        self._reject_locked_targets(filename)
        self._publish_temporary(filename, destination, temporary, held)

        # 2. Record the digest, then remove the held previous bytes.
        try:
            self._publish_digest(filename, content_sha256, publication_id)
        except RawStorageError:
            self._restore_held_file(filename)
            raise
        if held is not None:
            held.unlink(missing_ok=True)
        return destination

    def _require_matching_digest(
        self,
        filename: str,
        temporary: Path,
        content_sha256: str,
    ) -> None:
        """Reject a temporary copy whose bytes do not match the digest.

        Args:
            filename: Raw file name, used in the error message.
            temporary: Copied bytes to hash.
            content_sha256: Lowercase SHA-256 hex digest the copy must match.

        Returns:
            None.

        Raises:
            RawStorageError: The copied bytes do not match the digest.
        """
        if _digest_matches(_sha256_hex(temporary), content_sha256):
            return
        temporary.unlink(missing_ok=True)
        message = f"Could not store raw file {filename}"
        raise RawStorageError(message)

    def _publish_digest(
        self,
        filename: str,
        content_sha256: str,
        publication_id: str,
    ) -> None:
        """Record one file's digest for one publication.

        Args:
            filename: Raw file name.
            content_sha256: Lowercase SHA-256 hex digest.
            publication_id: Publication the digest belongs to.

        Returns:
            None.

        Raises:
            RawStorageError: The digest record could not be written.

        Flow:
            1. Write the record beside the raw directory.
            2. Publish that record under the raw file name.
        """
        # 1. Write the record beside the raw directory.
        record = self._write_digest_temporary(filename, content_sha256, publication_id)

        # 2. Publish that record under the raw file name.
        self._publish_digest_record(filename, record)

    def _clear_publication_record(self, filename: str) -> None:
        """Remove one file's publication record.

        Args:
            filename: Raw file name whose record is removed.

        Returns:
            None.

        Raises:
            RawStorageError: The digest directory is a symbolic link, or the
                record could not be removed.
        """
        record = self._digest_path(filename)
        self._reject_digest_symlink()
        try:
            record.unlink(missing_ok=True)
        except OSError as exc:
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc

    def _regular_destination(self, filename: str) -> Path:
        """Return the raw path when it is not a symbolic link.

        Args:
            filename: Raw file name.

        Returns:
            The path of that file inside the raw directory.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link.
        """
        require_raw_filename(filename)
        path = self._raw_dir / filename
        if path.is_symlink():
            message = f"Raw file {filename} is a symbolic link"
            raise RawStorageError(message)
        return path

    def _reject_locked_targets(self, filename: str) -> None:
        """Reject a destination or digest directory that is a symbolic link.

        Args:
            filename: Raw file name, used in the destination error.

        Returns:
            None.

        Raises:
            RawStorageError: The destination or digest directory is a symbolic link.
        """
        destination = self._raw_dir / filename
        if destination.is_symlink():
            message = f"Raw file {filename} is a symbolic link"
            raise RawStorageError(message)
        self._reject_digest_symlink()

    def _hold_previous_bytes(self, filename: str, destination: Path) -> Path | None:
        """Link a regular destination aside before it is replaced.

        Args:
            filename: Raw file name being replaced.
            destination: Current raw path.

        Returns:
            The held path, or None when no regular file is present.

        Raises:
            RawStorageError: The previous bytes could not be held.

        Flow:
            1. Leave an absent destination without a hold.
            2. Link the regular file into the hold directory.
        """
        # 1. Leave an absent destination without a hold.
        if not destination.is_file():
            return None

        # 2. Link the regular file into the hold directory.
        self._prepare_hold_files_directory()
        held = self._hold_path(filename)
        try:
            os.link(destination, held, follow_symlinks=False)
        except OSError as exc:
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc
        return held

    def _restore_held_file(self, filename: str) -> None:
        """Put a held copy back when its raw file is missing.

        Args:
            filename: Raw file name.

        Returns:
            None.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination or hold is a symbolic link, or the
                held copy could not be restored.

        Flow:
            1. Leave the raw file in place when it is already present.
            2. Restore the held copy only when the digest record can be read.
        """
        # 1. Leave the raw file in place when it is already present.
        require_raw_filename(filename)
        self._reject_locked_targets(filename)
        self._reject_hold_files_symlink()
        destination = self._raw_dir / filename
        if destination.exists() or destination.is_symlink():
            return

        # 2. Restore the held copy only when the digest record can be read.
        self._read_digest_record(filename)
        hold = self._hold_path(filename)
        if hold.is_symlink() or not hold.is_file():
            return
        try:
            os.link(hold, destination, follow_symlinks=False)
            hold.unlink()
        except OSError as exc:
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc

    def _copy_to_temporary(self, filename: str, source_file: Path) -> Path:
        """Copy source bytes into a new temporary file in the raw directory.

        Args:
            filename: Raw file name, used in the error message.
            source_file: Local file whose bytes are copied.

        Returns:
            The temporary path that holds those bytes.

        Raises:
            RawStorageError: The temporary file could not be created or filled.
        """
        self._reject_raw_directory_symlink()
        try:
            return self._create_exclusive_temporary(source_file)
        except OSError as exc:
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc

    def _create_exclusive_temporary(self, source_file: Path) -> Path:
        """Return a new file that holds a copy of the source bytes.

        Args:
            source_file: Local file whose bytes are copied.

        Returns:
            The temporary path.

        Raises:
            OSError: The raw directory, temporary file, or copy failed.
        """
        self._prepare_raw_directory()
        descriptor = _exclusive_temporary(self._raw_dir)
        temporary = descriptor[0]
        try:
            with source_file.open("rb") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
        except OSError:
            os.close(descriptor[1])
            temporary.unlink(missing_ok=True)
            raise
        os.close(descriptor[1])
        return temporary

    def _publish_temporary(
        self,
        filename: str,
        destination: Path,
        temporary: Path,
        held: Path | None,
    ) -> None:
        """Publish a temporary file without leaving it behind on failure.

        Args:
            filename: Raw file name, used in the error message.
            destination: Raw path that receives the copy.
            temporary: Temporary file to publish.
            held: Previous bytes to restore when publication fails.

        Returns:
            None.

        Raises:
            RawStorageError: The temporary file could not be published.
        """
        try:
            temporary.replace(destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            self._restore_after_failed_replace(filename, destination, held)
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc

    def _restore_after_failed_replace(
        self,
        filename: str,
        destination: Path,
        held: Path | None,
    ) -> None:
        """Restore held bytes when publication removed the destination.

        Args:
            filename: Raw file name.
            destination: Raw path that should hold the file.
            held: Previous bytes, or None when there was nothing to hold.

        Returns:
            None.
        """
        if held is None or destination.exists():
            return
        try:
            os.link(held, destination, follow_symlinks=False)
        except OSError as exc:
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc

    @contextmanager
    def _acquire_publish_lock(self) -> Iterator[None]:
        """Hold the publication lock for one store instance.

        Yields:
            None while the lock is held.

        Raises:
            RawStorageError: The lock could not be acquired.

        Flow:
            1. Reuse the lock already held by this instance.
            2. Acquire the lock and release it when the outermost hold ends.
        """
        # 1. Reuse the lock already held by this instance.
        if self._publish_lock_depth > 0:
            self._publish_lock_depth += 1
            try:
                yield
            finally:
                self._publish_lock_depth -= 1
            return

        # 2. Acquire the lock and release it when the outermost hold ends.
        descriptor = self._open_publish_lock()
        self._publish_lock_depth = 1
        try:
            yield
        finally:
            self._publish_lock_depth = 0
            os.close(descriptor)

    def _open_publish_lock(self) -> int:
        """Create and lock the publication lock file.

        Returns:
            The locked file descriptor.

        Raises:
            RawStorageError: The lock file could not be created or locked.
        """
        self._reject_holds_symlink()
        self._reject_raw_directory_symlink()
        try:
            lock_path = self._prepare_holds_directory() / _PUBLISH_LOCK_NAME
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            message = "Could not lock raw publication"
            raise RawStorageError(message) from exc
        return descriptor

    def _prepare_raw_directory(self) -> None:
        """Create the raw directory when it does not already exist.

        Returns:
            None.

        Raises:
            OSError: The raw directory could not be created.
        """
        self._reject_raw_directory_symlink()
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_holds_directory(self) -> Path:
        """Create the hold directory when it is not a symbolic link.

        Returns:
            The hold directory.

        Raises:
            RawStorageError: The hold directory could not be prepared.
        """
        self._reject_holds_symlink()
        directory = self._raw_dir / _HOLDS_DIRECTORY_NAME
        try:
            self._prepare_raw_directory()
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            message = "Could not prepare the raw directory"
            raise RawStorageError(message) from exc
        if directory.is_symlink():
            message = "Hold directory is a symbolic link"
            raise RawStorageError(message)
        return directory

    def _prepare_hold_files_directory(self) -> Path:
        """Create the directory that stores held raw files.

        Returns:
            The hold-files directory.

        Raises:
            RawStorageError: The hold-files directory could not be prepared.
        """
        self._prepare_holds_directory()
        directory = self._reject_hold_files_symlink()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            message = "Could not prepare the raw directory"
            raise RawStorageError(message) from exc
        if directory.is_symlink():
            message = "Hold directory is a symbolic link"
            raise RawStorageError(message)
        return directory

    def _reject_raw_directory_symlink(self) -> None:
        """Reject a raw directory or ancestor that is a symbolic link.

        Returns:
            None.

        Raises:
            RawStorageError: The raw directory or an existing ancestor is a
                symbolic link.
        """
        if self._raw_dir.is_symlink() or _ancestor_is_symlink(self._raw_dir):
            message = "Raw directory is a symbolic link"
            raise RawStorageError(message)

    def _reject_holds_symlink(self) -> None:
        """Reject a hold directory that is a symbolic link.

        Returns:
            None.

        Raises:
            RawStorageError: The hold directory is a symbolic link.
        """
        if (self._raw_dir / _HOLDS_DIRECTORY_NAME).is_symlink():
            message = "Hold directory is a symbolic link"
            raise RawStorageError(message)

    def _reject_hold_files_symlink(self) -> Path:
        """Reject a hold-files directory that is a symbolic link.

        Returns:
            The hold-files directory.

        Raises:
            RawStorageError: The hold directory or hold-files directory is a
                symbolic link.
        """
        self._reject_holds_symlink()
        directory = self._raw_dir / _HOLDS_DIRECTORY_NAME / _HOLD_FILES_DIRECTORY_NAME
        if directory.is_symlink():
            message = "Hold directory is a symbolic link"
            raise RawStorageError(message)
        return directory

    def _reject_digest_symlink(self) -> None:
        """Reject a digest directory that is a symbolic link.

        Returns:
            None.

        Raises:
            RawStorageError: The digest directory is a symbolic link.
        """
        if (self._raw_dir / _DIGESTS_DIRECTORY_NAME).is_symlink():
            message = "Digest directory is a symbolic link"
            raise RawStorageError(message)

    def _hold_path(self, filename: str) -> Path:
        """Return the hold path for one raw file.

        Args:
            filename: Raw file name.

        Returns:
            The hold path inside the hold-files directory.
        """
        return (
            self._raw_dir
            / _HOLDS_DIRECTORY_NAME
            / _HOLD_FILES_DIRECTORY_NAME
            / filename
        )

    def _digest_path(self, filename: str) -> Path:
        """Return the digest-record path for one raw file.

        Args:
            filename: Raw file name.

        Returns:
            The digest-record path inside the digest directory.
        """
        return self._raw_dir / _DIGESTS_DIRECTORY_NAME / filename

    def _write_digest_temporary(
        self,
        filename: str,
        content_sha256: str,
        publication_id: str,
    ) -> Path:
        """Write one digest record to a new temporary file.

        Args:
            filename: Raw file name, used in the error message.
            content_sha256: Lowercase SHA-256 hex digest.
            publication_id: Publication the digest belongs to.

        Returns:
            The temporary record path.

        Raises:
            RawStorageError: The record could not be written.
        """
        self._reject_digest_symlink()
        payload = json.dumps(
            {"publication_id": publication_id, "sha256": content_sha256},
        )
        try:
            self._prepare_raw_directory()
            digests = self._raw_dir / _DIGESTS_DIRECTORY_NAME
            digests.mkdir(parents=True, exist_ok=True)
            descriptor = _exclusive_temporary(digests)
        except OSError as exc:
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc
        temporary = descriptor[0]
        try:
            temporary.write_text(payload, encoding="utf-8")
        except OSError as exc:
            os.close(descriptor[1])
            temporary.unlink(missing_ok=True)
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc
        os.close(descriptor[1])
        return temporary

    def _publish_digest_record(self, filename: str, temporary: Path) -> None:
        """Publish a temporary digest record under the raw file name.

        Args:
            filename: Raw file name.
            temporary: Temporary record to publish.

        Returns:
            None.

        Raises:
            RawStorageError: The record could not be published.
        """
        self._reject_digest_symlink()
        try:
            temporary.replace(self._digest_path(filename))
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc

    def _read_digest_record(self, filename: str) -> tuple[str, str] | None:
        """Read one publication id and digest.

        Args:
            filename: Raw file name.

        Returns:
            The publication id and lowercase SHA-256, or None when no record exists.

        Raises:
            RawStorageError: The digest directory is a symbolic link, or the
                record cannot be parsed.
        """
        record = self._digest_path(filename)
        self._reject_digest_symlink()
        if record.is_symlink():
            message = f"Could not read publication record for {filename}"
            raise RawStorageError(message)
        try:
            text = record.read_text(encoding="utf-8") if record.is_file() else None
        except OSError as exc:
            message = f"Could not read publication record for {filename}"
            raise RawStorageError(message) from exc
        if text is None:
            return None
        return _parse_digest_record(filename, text)


def _exclusive_temporary(directory: Path) -> tuple[Path, int]:
    """Create a new empty file that does not replace an existing name.

    Args:
        directory: Directory that receives the temporary file.

    Returns:
        The temporary path and its open file descriptor.

    Raises:
        OSError: The temporary file could not be created.
    """
    descriptor, name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=directory)
    return Path(name), descriptor


def _ancestor_is_symlink(path: Path) -> bool:
    """Return whether an existing ancestor of a path is a symbolic link.

    Args:
        path: Path whose ancestors are checked. The path itself may be absent.

    Returns:
        True when an existing ancestor is a symbolic link.
    """
    current = path.parent
    while True:
        if current.is_symlink():
            return True
        if current.exists() or current.parent == current:
            return False
        current = current.parent


def _sha256_hex(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of one file.

    Args:
        path: File to hash.

    Returns:
        The lowercase SHA-256 hex digest.

    Raises:
        RawStorageError: The file could not be read.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        message = f"Could not hash raw file {path.name}"
        raise RawStorageError(message) from exc
    return digest.hexdigest()


def _digest_matches(actual: str, expected: str) -> bool:
    """Return whether two SHA-256 hex digests are the same.

    Args:
        actual: Digest computed from stored bytes.
        expected: Digest supplied by the caller.

    Returns:
        True when both values are the same lowercase SHA-256 hex digest.
    """
    if len(expected) != _SHA256_HEX_LENGTH or expected.lower() != expected:
        return False
    return actual == expected


def _parse_digest_record(filename: str, text: str) -> tuple[str, str]:
    """Return a publication id and digest from one record.

    Args:
        filename: Raw file name, used in the error message.
        text: JSON record text.

    Returns:
        The publication id and lowercase SHA-256.

    Raises:
        RawStorageError: The record is not the expected JSON object.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        message = f"Could not read publication record for {filename}"
        raise RawStorageError(message) from exc
    if not isinstance(payload, dict):
        message = f"Could not read publication record for {filename}"
        raise RawStorageError(message)
    publication_id = payload.get("publication_id")
    digest = payload.get("sha256")
    if (
        not isinstance(publication_id, str)
        or not isinstance(digest, str)
        or not _digest_matches(digest, digest)
    ):
        message = f"Could not read publication record for {filename}"
        raise RawStorageError(message)
    return publication_id, digest
