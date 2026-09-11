"""Native NVFP4 dense projections against an independent dequantized reference."""

import pytest
import torch

from freetoken.kernel.triton.nvfp4_linear import (
    nvfp4_dense_linear,
    nvfp4_dense_linear_t,
    nvfp4_transpose_resident,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA or ROCm GPU")
@pytest.mark.parametrize("batch", [1, 4, 64])
@pytest.mark.parametrize("transposed", [False, True])
def test_nvfp4_dense_projection_matches_reference(batch, transposed):
    torch.manual_seed(19)
    n, k = 512, 256
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scale = (torch.rand(n, k // 16, device="cuda") + 0.25).to(torch.float8_e4m3fn)
    glob = (torch.rand(n, device="cuda") * 0.25 + 0.125).half()
    x = torch.randn(batch, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(n, k).long()
    weight = lut[codes] * scale.float().repeat_interleave(16, dim=1) * glob.float()[:, None]
    reference = (x.float() @ weight.T).bfloat16() + bias

    if transposed:
        weight_t, scale_t = nvfp4_transpose_resident(packed, scale)
        result = nvfp4_dense_linear_t(x, weight_t, scale_t, glob, bias)
    else:
        result = nvfp4_dense_linear(x, packed, scale, glob, bias)
    torch.testing.assert_close(result, reference, rtol=0.02, atol=0.02 * reference.abs().max().item())
