"""Filesystem adapter for unmodified raw dataset files."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from tiny_story.application.ports.raw_dataset_store import RawDatasetStore
from tiny_story.domain.entities.dataset_spec import require_raw_filename
from tiny_story.domain.errors import RawStorageError


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

    @property
    def directory(self) -> Path:
        """Directory that receives unmodified raw files.

        Returns:
            The raw directory.
        """
        return self._raw_dir

    def existing_file(self, filename: str) -> Path | None:
        """Return the stored file when a non-empty copy is already present.

        Args:
            filename: Raw file name.

        Returns:
            The raw path, or None when that file is absent or empty.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link.

        Flow:
            1. Reject an unsafe filename or a destination symbolic link.
            2. Return the path when a non-empty regular file is already stored.
        """
        # 1. Reject an unsafe filename or a destination symbolic link.
        path = self._regular_destination(filename)

        # 2. Return the path when a non-empty regular file is already stored.
        if path.is_file() and path.stat().st_size > 0:
            return path
        return None

    def store_unchanged(self, filename: str, source_file: Path) -> Path:
        """Copy acquired bytes into the raw directory.

        Args:
            filename: Raw file name.
            source_file: Local file whose bytes are copied.

        Returns:
            The raw path that now holds those bytes.

        Raises:
            InvalidRawFilenameError: The name would leave the raw directory.
            RawStorageError: The destination is a symbolic link, or the bytes
                could not be copied.

        Flow:
            1. Reject an unsafe filename or a destination symbolic link.
            2. Copy the bytes into a new exclusive temporary file.
            3. Publish the temporary file as the raw file.
        """
        # 1. Reject an unsafe filename or a destination symbolic link.
        destination = self._regular_destination(filename)

        # 2. Copy the bytes into a new exclusive temporary file.
        temporary = self._copy_to_temporary(filename, source_file)

        # 3. Publish the temporary file as the raw file.
        try:
            temporary.replace(destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc
        return destination

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

    def _copy_to_temporary(self, filename: str, source_file: Path) -> Path:
        """Copy source bytes into a new temporary file in the raw directory.

        Args:
            filename: Raw file name, used in the error message.
            source_file: Local file whose bytes are copied.

        Returns:
            The temporary path that holds those bytes.

        Raises:
            RawStorageError: The temporary file could not be created or filled.

        Flow:
            1. Create an exclusive temporary file in the raw directory.
            2. Copy the source bytes into it, and remove it if the copy fails.
        """
        # 1. Create an exclusive temporary file in the raw directory.
        try:
            temporary = self._create_exclusive_temporary()
        except OSError as exc:
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc

        # 2. Copy the source bytes into it, and remove it if the copy fails.
        try:
            shutil.copyfile(source_file, temporary)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            message = f"Could not store raw file {filename}"
            raise RawStorageError(message) from exc
        return temporary

    def _create_exclusive_temporary(self) -> Path:
        """Return a new empty file that does not replace an existing name.

        Returns:
            The temporary path.

        Raises:
            OSError: The raw directory or temporary file could not be created.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".",
            suffix=".tmp",
            dir=self._raw_dir,
        )
        os.close(descriptor)
        return Path(name)
