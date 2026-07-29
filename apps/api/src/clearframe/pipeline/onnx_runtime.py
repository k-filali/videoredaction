from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from numpy.typing import NDArray

PREFERRED_EXECUTION_PROVIDERS = (
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)
ONNX_RUNTIME_VERSION = ort.__version__


class OnnxNodeArgument(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> Sequence[int | str | None]: ...

    @property
    def type(self) -> str: ...


class OnnxSession(Protocol):
    def get_inputs(self) -> Sequence[OnnxNodeArgument]: ...

    def get_outputs(self) -> Sequence[OnnxNodeArgument]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[np.float32]],
    ) -> Sequence[object]: ...


SessionFactory = Callable[[str, tuple[str, ...]], OnnxSession]
ProviderDiscovery = Callable[[], Sequence[str]]


def discover_execution_providers() -> Sequence[str]:
    return cast(Sequence[str], ort.get_available_providers())


def select_execution_providers(available: Sequence[str]) -> tuple[str, ...]:
    available_set = set(available)
    providers = tuple(
        provider
        for provider in PREFERRED_EXECUTION_PROVIDERS
        if provider in available_set
    )
    if not providers:
        raise RuntimeError("ONNX Runtime has no supported CUDA or CPU execution provider")
    return providers


def create_onnx_session(
    model_path: str,
    providers: tuple[str, ...],
) -> OnnxSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    return cast(
        OnnxSession,
        ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=list(providers),
        ),
    )


def create_session_with_cpu_fallback(
    model_path: str,
    providers: tuple[str, ...],
    factory: SessionFactory,
) -> tuple[OnnxSession, tuple[str, ...]]:
    try:
        return factory(model_path, providers), providers
    except Exception as preferred_error:
        cpu_providers = ("CPUExecutionProvider",)
        if (
            providers[0] != "CUDAExecutionProvider"
            or "CPUExecutionProvider" not in providers
        ):
            raise
        try:
            return factory(model_path, cpu_providers), cpu_providers
        except Exception as fallback_error:
            raise fallback_error from preferred_error


def active_execution_provider(
    session: OnnxSession,
    requested: Sequence[str],
) -> str:
    active = tuple(session.get_providers())
    provider = next((name for name in active if name in requested), None)
    if provider is None:
        raise RuntimeError("ONNX Runtime did not activate a requested execution provider")
    return provider


def provider_device(provider: str) -> str:
    if provider == "CUDAExecutionProvider":
        return "cuda"
    if provider == "CPUExecutionProvider":
        return "cpu"
    raise ValueError(f"unsupported ONNX Runtime execution provider: {provider}")
