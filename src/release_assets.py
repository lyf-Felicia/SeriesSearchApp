from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import quote, urlparse


RELEASE_REPOSITORY = "lyf-Felicia/SeriesSearchApp"
RELEASE_TAG = "1.0"
ALLOWED_DOWNLOAD_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
}
MAX_ARCHIVE_MEMBERS = 100_000
MAX_UNCOMPRESSED_BYTES = 12 * 1024**3
MAX_COMPRESSION_RATIO = 200
INTEGRITY_MARKER = ".release-sha256"


class AssetValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    destination: str
    size: int
    sha256: str
    extract_directory: str | None = None

    @property
    def url(self) -> str:
        repository = quote(RELEASE_REPOSITORY, safe="/")
        tag = quote(RELEASE_TAG, safe="")
        name = quote(self.name, safe="")
        return f"https://github.com/{repository}/releases/download/{tag}/{name}"


RELEASE_ASSETS = (
    ReleaseAsset(
        name="llm_summaries.json",
        destination="llm_summaries.json",
        size=8_559_436,
        sha256="a1d3e7ddfbf998099d01bb3469e7a225111157c5c5660965f2776c4950146805",
    ),
    ReleaseAsset(
        name="final.db",
        destination="database/final.db",
        size=533_716_992,
        sha256="e87ae1a4683750193482fa4c90c8ddcf1132972a42aace9433f6492e41268aeb",
    ),
    ReleaseAsset(
        name="qdrant_data.zip",
        destination="qdrant_data.zip",
        size=2_003_422_530,
        sha256="13bebd00a110ecfb5ffdb4592c5d3dbedee0d48529086a3b150fc24456d01143",
        extract_directory="qdrant_data",
    ),
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, asset: ReleaseAsset) -> None:
    if not path.is_file():
        raise AssetValidationError(f"Missing asset: {asset.name}")
    actual_size = path.stat().st_size
    if actual_size != asset.size:
        raise AssetValidationError(
            f"Unexpected size for {asset.name}: {actual_size} bytes"
        )
    if sha256_file(path) != asset.sha256:
        raise AssetValidationError(f"SHA-256 mismatch for {asset.name}")


def download_asset(
    asset: ReleaseAsset,
    data_directory: Path,
    progress: Callable[[int, int], None] | None = None,
    timeout: int = 60,
) -> Path:
    destination = data_directory / asset.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "SeriesSearchApp/1.0"},
    )

    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https" or final_url.hostname not in ALLOWED_DOWNLOAD_HOSTS:
                raise AssetValidationError("Release download redirected to an untrusted host")

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) != asset.size:
                raise AssetValidationError(
                    f"Unexpected Content-Length for {asset.name}: {content_length}"
                )

            file_descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{asset.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary_path = Path(raw_path)
            downloaded = 0
            with os.fdopen(file_descriptor, "wb") as output:
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > asset.size:
                        raise AssetValidationError(f"Asset exceeds declared size: {asset.name}")
                    output.write(chunk)
                    if progress:
                        progress(downloaded, asset.size)
                output.flush()
                os.fsync(output.fileno())

        validate_file(temporary_path, asset)
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validated_archive_members(archive: zipfile.ZipFile, expected_root: str):
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise AssetValidationError("Archive contains too many members")

    total_size = 0
    for member in members:
        path = PurePosixPath(member.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != expected_root
        ):
            raise AssetValidationError(f"Unsafe archive path: {member.filename}")

        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise AssetValidationError(f"Archive symlink is not allowed: {member.filename}")

        total_size += member.file_size
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise AssetValidationError("Archive expands beyond the configured limit")
        if (
            member.file_size > 0
            and member.compress_size > 0
            and member.file_size / member.compress_size > MAX_COMPRESSION_RATIO
        ):
            raise AssetValidationError(f"Suspicious compression ratio: {member.filename}")

    return members


def extract_qdrant_archive(
    archive_path: Path, data_directory: Path, archive_sha256: str | None = None
) -> Path:
    target = data_directory / "qdrant_data"
    extraction_root = Path(tempfile.mkdtemp(prefix=".qdrant-extract-", dir=data_directory))
    backup = data_directory / ".qdrant-backup"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _validated_archive_members(archive, target.name)
            archive.extractall(extraction_root, members)

        extracted = extraction_root / target.name
        if not (extracted / "meta.json").is_file():
            raise AssetValidationError("Qdrant archive is missing qdrant_data/meta.json")
        if archive_sha256:
            (extracted / INTEGRITY_MARKER).write_text(
                archive_sha256 + "\n", encoding="ascii"
            )

        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(extracted, target)
        except Exception:
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return target
    finally:
        shutil.rmtree(extraction_root, ignore_errors=True)


def ensure_release_assets(data_directory: Path) -> None:
    data_directory.mkdir(parents=True, exist_ok=True)
    for asset in RELEASE_ASSETS:
        destination = data_directory / asset.destination
        if asset.extract_directory:
            extracted_directory = data_directory / asset.extract_directory
            marker = extracted_directory / INTEGRITY_MARKER
            if (
                (extracted_directory / "meta.json").is_file()
                and marker.is_file()
                and marker.read_text(encoding="ascii").strip() == asset.sha256
            ):
                continue
        try:
            validate_file(destination, asset)
        except AssetValidationError:
            destination = download_asset(asset, data_directory)
        if asset.extract_directory:
            extract_qdrant_archive(destination, data_directory, asset.sha256)
            destination.unlink(missing_ok=True)