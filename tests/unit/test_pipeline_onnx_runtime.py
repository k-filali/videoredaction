import os
from typing import cast

import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from clearframe.pipeline.onnx_runtime import (
    OnnxSession,
    active_execution_provider,
    create_onnx_session,
    provider_device,
    select_execution_providers,
)


class _ProviderSession:
    def __init__(self, providers: tuple[str, ...]) -> None:
        self.providers = providers

    def get_providers(self) -> tuple[str, ...]:
        return self.providers


def test_provider_selection_prefers_cuda_with_cpu_fallback() -> None:
    assert select_execution_providers(
        (
            "AzureExecutionProvider",
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        )
    ) == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert select_execution_providers(("CPUExecutionProvider",)) == (
        "CPUExecutionProvider",
    )
    with pytest.raises(RuntimeError, match="no supported CUDA or CPU"):
        select_execution_providers(("AzureExecutionProvider",))


def test_active_provider_comes_from_created_session() -> None:
    requested = ("CUDAExecutionProvider", "CPUExecutionProvider")
    cuda_session = cast(
        OnnxSession,
        _ProviderSession(("CUDAExecutionProvider", "CPUExecutionProvider")),
    )
    fallback_session = cast(
        OnnxSession,
        _ProviderSession(("CPUExecutionProvider",)),
    )

    assert active_execution_provider(cuda_session, requested) == "CUDAExecutionProvider"
    assert active_execution_provider(fallback_session, requested) == "CPUExecutionProvider"
    assert provider_device("CUDAExecutionProvider") == "cuda"
    assert provider_device("CPUExecutionProvider") == "cpu"


def test_cpu_session_options_limit_thread_oversubscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_session(
        model_path: str,
        *,
        sess_options: ort.SessionOptions,
        providers: list[str],
    ) -> object:
        captured["model_path"] = model_path
        captured["options"] = sess_options
        captured["providers"] = providers
        return sentinel

    monkeypatch.setattr(ort, "InferenceSession", fake_session)

    session = create_onnx_session("model.onnx", ("CPUExecutionProvider",))

    options = cast(ort.SessionOptions, captured["options"])
    assert session is sentinel
    assert captured["model_path"] == "model.onnx"
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert options.inter_op_num_threads == 1
    assert options.intra_op_num_threads == min(4, os.cpu_count() or 1)
