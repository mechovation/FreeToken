"""ROCm must never select NVIDIA packages or pass nvcc-only compiler flags."""

import pytest
import torch

from freetoken.kernel import backend
from freetoken.kernel.utils import _cuda_cflags
from freetoken.utils import arch


@pytest.mark.parametrize("hip", [None, "7.2.0"])
def test_optional_cuda_packages_are_disabled_on_rocm(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(backend, "_importable", lambda name: True)
    backend.is_flashinfer_installed.cache_clear()
    backend.is_sgl_kernel_installed.cache_clear()
    try:
        assert backend.is_flashinfer_installed() is (hip is None)
        assert backend.is_sgl_kernel_installed() is (hip is None)
    finally:
        backend.is_flashinfer_installed.cache_clear()
        backend.is_sgl_kernel_installed.cache_clear()


def test_hip_compiler_does_not_inherit_nvcc_flags(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "8.6 12.0")
    assert _cuda_cflags(["-DTEST=1"]) == ["-std=c++20", "-O3", "-DTEST=1"]


def test_amd_capability_is_not_nvidia_sm120(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    arch._get_torch_cuda_version.cache_clear()
    try:
        assert arch.is_rocm()
        assert not arch.is_sm90_supported()
        assert not arch.is_sm100_supported()
    finally:
        arch._get_torch_cuda_version.cache_clear()


def test_rocm_e4m3_wrapper_uses_portable_kernel_convention(monkeypatch):
    from freetoken.kernel.triton import e4m3_compat

    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 0))
    monkeypatch.setattr(e4m3_compat, "_native", None)
    scale = torch.ones(16).to(torch.float8_e4m3fn)
    view = e4m3_compat.e4m3_kernel_view(scale)
    assert not e4m3_compat.e4m3_native()
    assert e4m3_compat.e4m3_act_dtype() == torch.bfloat16
    assert view.dtype == torch.uint8
    assert view.data_ptr() == scale.data_ptr()
