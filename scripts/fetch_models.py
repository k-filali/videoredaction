from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "configs" / "models" / "weights"
DEFAULT_TIMEOUT_SECONDS = 120.0


class ModelIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    key: str
    filename: str
    url: str
    size_bytes: int
    sha256: str
    license_spdx: str
    license_url: str


FACE_REVISION = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"

ARTIFACTS = {
    "face": ModelArtifact(
        key="face",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://huggingface.co/opencv/face_detection_yunet/resolve/"
            f"{FACE_REVISION}/face_detection_yunet_2023mar.onnx"
        ),
        size_bytes=232_589,
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        license_spdx="MIT",
        license_url=(
            "https://huggingface.co/opencv/face_detection_yunet/blob/"
            f"{FACE_REVISION}/LICENSE"
        ),
    ),
    "plate": ModelArtifact(
        key="plate",
        filename="license_plate_detection_yolov9s_608.onnx",
        url=(
            "https://github.com/ankandrew/open-image-models/releases/download/"
            "assets/yolo-v9-s-608-license-plates-end2end.onnx"
        ),
        size_bytes=28_612_350,
        sha256="2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a",
        license_spdx="MIT",
        license_url="https://github.com/ankandrew/open-image-models/blob/main/LICENSE",
    ),
}

Downloader = Callable[[str, Path, float], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, artifact: ModelArtifact) -> None:
    if not path.is_file():
        raise ModelIntegrityError(f"model file is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != artifact.size_bytes:
        raise ModelIntegrityError(
            f"{artifact.filename} has {actual_size} bytes; expected {artifact.size_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != artifact.sha256:
        raise ModelIntegrityError(
            f"{artifact.filename} has SHA-256 {actual_sha256}; expected {artifact.sha256}"
        )


def _download(url: str, destination: Path, timeout_seconds: float) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ClearFrame-model-fetcher/1.0"},
    )
    with (
        urllib.request.urlopen(request, timeout=timeout_seconds) as response,
        destination.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def fetch_artifact(
    artifact: ModelArtifact,
    output_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    downloader: Downloader = _download,
) -> tuple[Path, bool]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / artifact.filename
    try:
        verify_artifact(destination, artifact)
    except ModelIntegrityError:
        pass
    else:
        return destination, False

    with tempfile.NamedTemporaryFile(
        dir=output_dir,
        prefix=f".{artifact.filename}.",
        suffix=".part",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        downloader(artifact.url, temporary_path, timeout_seconds)
        verify_artifact(temporary_path, artifact)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination, True


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch verified detector models")
    parser.add_argument(
        "--model",
        choices=("all", *ARTIFACTS),
        default="all",
        help="model to fetch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="destination for model weights",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="network timeout for each model",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selected = ARTIFACTS.values() if args.model == "all" else (ARTIFACTS[args.model],)
    for artifact in selected:
        path, downloaded = fetch_artifact(
            artifact,
            args.output_dir,
            timeout_seconds=args.timeout_seconds,
        )
        action = "downloaded" if downloaded else "verified"
        print(f"{artifact.key}: {action} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
