from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from clearframe.model_registry import (
    DEFAULT_REGISTRY_PATH,
    ModelRegistryError,
    load_model_registry,
)


def _write_registry(path: Path, model: dict[str, object]) -> Path:
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "models": [model]}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _weighted_model(weight_path: str | None, sha256: str | None) -> dict[str, object]:
    return {
        "id": "test-onnx",
        "display_name": "Test ONNX detector",
        "adapter": "onnx",
        "model_version": "1.0.0",
        "adapter_version": "1.0.0",
        "enabled": True,
        "deployment_allowed": False,
        "research_only": True,
        "supported_classes": ["license_plate"],
        "thresholds": {
            "confidence": 0.5,
            "nms_iou": 0.4,
            "min_size_pixels": 16,
        },
        "weights": {
            "source": "local",
            "path": weight_path,
            "sha256": sha256,
        },
        "license": {
            "spdx_id": "MIT",
            "name": "MIT License",
            "url": None,
        },
    }


def test_default_registry_is_fail_closed_and_fingerprinted() -> None:
    registry = load_model_registry()

    assert DEFAULT_REGISTRY_PATH.is_file()
    assert [model.id for model in registry.enabled_models] == ["mock-plate-v1"]
    assert len(registry.config_fingerprint) == 64
    assert registry.get("opencv-haar-face").enabled is False
    assert registry.get("yunet-face-local").weights.path is None


def test_fingerprint_is_semantic_and_independent_of_model_order(tmp_path: Path) -> None:
    source = yaml.safe_load(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    source["models"].reverse()
    second.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")

    assert load_model_registry(first).config_fingerprint == load_model_registry(
        second
    ).config_fingerprint


def test_enabled_local_weights_require_matching_sha256(tmp_path: Path) -> None:
    weight_path = tmp_path / "model.onnx"
    weight_path.write_bytes(b"known model bytes")
    expected = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        _weighted_model(weight_path.name, expected),
    )

    registry = load_model_registry(registry_path)
    assert registry.get("test-onnx").weights.sha256 == expected

    weight_path.write_bytes(b"tampered")
    with pytest.raises(ModelRegistryError, match="SHA-256 mismatch"):
        load_model_registry(registry_path)


@pytest.mark.parametrize(
    ("weight_path", "sha256", "message"),
    [
        (None, "0" * 64, "weight path"),
        ("model.onnx", None, "SHA-256"),
    ],
)
def test_enabled_weighted_model_rejects_incomplete_provenance(
    tmp_path: Path,
    weight_path: str | None,
    sha256: str | None,
    message: str,
) -> None:
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        _weighted_model(weight_path, sha256),
    )

    with pytest.raises(ValidationError, match=message):
        load_model_registry(registry_path)


def test_registry_rejects_remote_weights_and_unknown_fields(tmp_path: Path) -> None:
    remote = _weighted_model("https://example.test/model.onnx", "0" * 64)
    remote_path = _write_registry(tmp_path / "remote.yaml", remote)
    with pytest.raises(ValidationError, match="remote weight locations"):
        load_model_registry(remote_path)

    unknown = _weighted_model("model.onnx", "0" * 64)
    unknown["download_url"] = "https://example.test/model.onnx"
    unknown_path = _write_registry(tmp_path / "unknown.yaml", unknown)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_model_registry(unknown_path)


def test_disabled_future_slot_must_be_complete_before_enablement(tmp_path: Path) -> None:
    model = _weighted_model(None, None)
    model["enabled"] = False
    model["license"] = None
    registry_path = _write_registry(tmp_path / "registry.yaml", model)
    assert load_model_registry(registry_path).enabled_models == ()

    model["enabled"] = True
    _write_registry(registry_path, model)
    with pytest.raises(ValidationError, match="explicit license"):
        load_model_registry(registry_path)
