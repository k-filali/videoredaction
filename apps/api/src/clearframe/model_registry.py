from __future__ import annotations

import hashlib
import importlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Self, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "models" / "registry.yaml"
)
SHA256_LENGTH = 64


class ModelRegistryError(ValueError):
    """Raised when a registry cannot be used safely."""


class DetectorClass(StrEnum):
    FACE = "face"
    LICENSE_PLATE = "license_plate"


class AdapterKind(StrEnum):
    DETERMINISTIC_MOCK = "deterministic_mock"
    OPENCV_CASCADE = "opencv_cascade"
    OPENCV_YUNET = "opencv_yunet"
    ONNX = "onnx"


class WeightSource(StrEnum):
    NONE = "none"
    LOCAL = "local"
    OPENCV_DATA = "opencv_data"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelLicense(StrictModel):
    spdx_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    url: HttpUrl | None = None


class DetectionThresholds(StrictModel):
    confidence: float = Field(ge=0.0, le=1.0)
    nms_iou: float = Field(ge=0.0, le=1.0)
    min_size_pixels: int = Field(gt=0)


class WeightSpec(StrictModel):
    source: WeightSource
    path: str | None = None
    sha256: str | None = Field(default=None, min_length=SHA256_LENGTH, max_length=SHA256_LENGTH)

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.sha256 is not None:
            try:
                bytes.fromhex(self.sha256)
            except ValueError as exc:
                raise ValueError("weights.sha256 must be lowercase hexadecimal") from exc
            if self.sha256 != self.sha256.lower():
                raise ValueError("weights.sha256 must be lowercase hexadecimal")

        if self.source is WeightSource.NONE and (self.path is not None or self.sha256 is not None):
            raise ValueError("weight-free models cannot declare a path or SHA-256")
        if self.source is WeightSource.OPENCV_DATA:
            if self.path is None:
                raise ValueError("OpenCV data weights require a cascade filename")
            path = Path(self.path)
            if path.name != self.path or path.suffix.lower() != ".xml":
                raise ValueError("OpenCV data weights must be a single XML filename")
        if self.source is WeightSource.LOCAL and self.path is not None:
            if "://" in self.path:
                raise ValueError("remote weight locations are not supported")
            if not self.path.strip():
                raise ValueError("local weight path cannot be blank")
        return self


class ModelEntry(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9]+(?:[a-z0-9-]*[a-z0-9])?$")
    display_name: str = Field(min_length=1)
    adapter: AdapterKind
    model_version: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    enabled: bool
    deployment_allowed: bool
    research_only: bool
    supported_classes: frozenset[DetectorClass] = Field(min_length=1)
    thresholds: DetectionThresholds
    weights: WeightSpec
    license: ModelLicense | None

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.deployment_allowed and self.research_only:
            raise ValueError("research-only models cannot be deployment allowed")
        if self.adapter is AdapterKind.DETERMINISTIC_MOCK:
            if self.weights.source is not WeightSource.NONE:
                raise ValueError("the deterministic mock cannot use weights")
        elif self.weights.source is WeightSource.NONE:
            raise ValueError("non-mock models must declare a weight source")

        if self.enabled:
            if self.license is None:
                raise ValueError("enabled models require an explicit license")
            if self.weights.source is not WeightSource.NONE:
                if self.weights.path is None:
                    raise ValueError("enabled weighted models require a weight path")
                if self.weights.sha256 is None:
                    raise ValueError("enabled weighted models require a SHA-256")
        return self


class ModelRegistry(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    models: tuple[ModelEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        return self

    def get(self, model_id: str) -> ModelEntry:
        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(model_id)

    @property
    def enabled_models(self) -> tuple[ModelEntry, ...]:
        return tuple(model for model in self.models if model.enabled)

    @property
    def config_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        models = cast(list[dict[str, object]], payload["models"])
        for model in models:
            classes = cast(list[str], model["supported_classes"])
            model["supported_classes"] = sorted(classes)
        payload["models"] = sorted(models, key=lambda model: cast(str, model["id"]))
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opencv_data_root() -> Path:
    try:
        cv2 = importlib.import_module("cv2")
        root = cv2.data.haarcascades
    except (ImportError, AttributeError) as exc:
        raise ModelRegistryError("OpenCV data directory is unavailable") from exc
    return Path(cast(str, root)).resolve()


def resolve_weight_path(
    model: ModelEntry,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    weights_root: Path | None = None,
) -> Path | None:
    if model.weights.source is WeightSource.NONE:
        return None
    if model.weights.path is None:
        raise ModelRegistryError(f"{model.id}: weight path is not configured")

    if model.weights.source is WeightSource.OPENCV_DATA:
        root = _opencv_data_root()
        path = (root / model.weights.path).resolve()
    else:
        configured = Path(model.weights.path)
        if configured.is_absolute():
            path = configured.resolve()
        else:
            root = (weights_root or registry_path.resolve().parent).resolve()
            path = (root / configured).resolve()
            if not path.is_relative_to(root):
                raise ModelRegistryError(f"{model.id}: weight path escapes its configured root")
    return path


def validate_enabled_weights(
    registry: ModelRegistry,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    weights_root: Path | None = None,
) -> None:
    for model in registry.enabled_models:
        path = resolve_weight_path(
            model,
            registry_path=registry_path,
            weights_root=weights_root,
        )
        if path is None:
            continue
        if not path.is_file():
            raise ModelRegistryError(f"{model.id}: weight file does not exist: {path}")
        expected = model.weights.sha256
        if expected is None:
            raise ModelRegistryError(f"{model.id}: enabled weighted models require a SHA-256")
        actual = _sha256(path)
        if actual != expected:
            raise ModelRegistryError(
                f"{model.id}: weight SHA-256 mismatch (expected {expected}, got {actual})"
            )


def load_model_registry(
    path: Path | str = DEFAULT_REGISTRY_PATH,
    *,
    weights_root: Path | None = None,
    verify_enabled_weights: bool = True,
) -> ModelRegistry:
    registry_path = Path(path).resolve()
    if not registry_path.is_file():
        raise ModelRegistryError(f"model registry does not exist: {registry_path}")

    try:
        raw = cast(object, yaml.safe_load(registry_path.read_text(encoding="utf-8")))
    except yaml.YAMLError as exc:
        raise ModelRegistryError(f"invalid model registry YAML: {registry_path}") from exc
    if not isinstance(raw, dict):
        raise ModelRegistryError("model registry root must be a mapping")

    registry = ModelRegistry.model_validate(raw)
    if verify_enabled_weights:
        validate_enabled_weights(
            registry,
            registry_path=registry_path,
            weights_root=weights_root,
        )
    return registry
