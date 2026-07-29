from pathlib import Path

import pytest

from clearframe.storage import LocalStorage, StorageSecurityError, sanitize_filename


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("../../private.mp4", "private.mp4"),
        (r"C:\fakepath\incident 01.MP4", "incident 01.mp4"),
        ("résumé?.mov", "r_sum_.mov"),
        ("\x00", "video"),
    ],
)
def test_filename_sanitization(source: str, expected: str) -> None:
    assert sanitize_filename(source) == expected


def test_storage_uris_cannot_escape_root(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "storage")

    with pytest.raises(StorageSecurityError):
        storage.path_for("../outside.mp4")
    with pytest.raises(StorageSecurityError):
        storage.path_for("/absolute/path.mp4")

    safe = storage.prepare("proxies/video-id/proxy.mp4")
    assert safe.is_relative_to(storage.root)
    assert safe.parent.is_dir()

