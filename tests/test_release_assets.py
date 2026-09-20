import hashlib
import io
import stat
import zipfile
from pathlib import Path

import pytest

from src.release_assets import (
    INTEGRITY_MARKER,
    AssetValidationError,
    ReleaseAsset,
    download_asset,
    extract_qdrant_archive,
    validate_file,
)


def test_validate_file_rejects_hash_mismatch(tmp_path: Path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"expected size")
    asset = ReleaseAsset(
        name="asset.bin",
        destination="asset.bin",
        size=path.stat().st_size,
        sha256=hashlib.sha256(b"different content").hexdigest(),
    )

    with pytest.raises(AssetValidationError, match="SHA-256 mismatch"):
        validate_file(path, asset)


def test_download_asset_validates_then_replaces(monkeypatch, tmp_path: Path):
    payload = b"verified payload"
    asset = ReleaseAsset(
        name="asset.bin",
        destination="nested/asset.bin",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def geturl(self):
            return asset.url

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response(payload))

    destination = download_asset(asset, tmp_path)

    assert destination.read_bytes() == payload
    assert not list(destination.parent.glob("*.tmp"))


def test_extract_rejects_path_traversal(tmp_path: Path):
    archive_path = tmp_path / "qdrant.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("qdrant_data/meta.json", "{}")
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(AssetValidationError, match="Unsafe archive path"):
        extract_qdrant_archive(archive_path, tmp_path)

    assert not (tmp_path.parent / "outside.txt").exists()


def test_extract_rejects_symlink(tmp_path: Path):
    archive_path = tmp_path / "qdrant.zip"
    symlink = zipfile.ZipInfo("qdrant_data/link")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("qdrant_data/meta.json", "{}")
        archive.writestr(symlink, "target")

    with pytest.raises(AssetValidationError, match="symlink"):
        extract_qdrant_archive(archive_path, tmp_path)


def test_extract_replaces_target_after_validation(tmp_path: Path):
    target = tmp_path / "qdrant_data"
    target.mkdir()
    (target / "meta.json").write_text('{"old": true}', encoding="utf-8")
    archive_path = tmp_path / "qdrant.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("qdrant_data/meta.json", '{"new": true}')
        archive.writestr("qdrant_data/collection/data", "payload")

    extract_qdrant_archive(archive_path, tmp_path, "expected-digest")

    assert (target / "meta.json").read_text(encoding="utf-8") == '{"new": true}'
    assert (target / "collection/data").read_text(encoding="utf-8") == "payload"
    assert (target / INTEGRITY_MARKER).read_text(encoding="ascii").strip() == "expected-digest"