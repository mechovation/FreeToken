"""Exercise the ROCm runtime, native JIT kernels, and Triton without model weights."""

import torch
import torch.nn.functional as F

from freetoken.kernel.index import indexing
from freetoken.kernel.fast_index_copy import fast_index_copy_jit
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.kernel.radix import fast_compare_key
from freetoken.kernel.store import store_cache
from freetoken.kernel.triton.activation import gelu_tanh_and_mul, silu_and_mul
from freetoken.kernel.triton.norm import rmsnorm


def main():
    assert torch.version.hip, "Install a ROCm build of PyTorch"
    assert torch.cuda.is_available(), "Pass /dev/kfd and /dev/dri into the container"
    print(f"torch={torch.__version__}, HIP={torch.version.hip}", flush=True)
    print(torch.cuda.get_device_properties(0), flush=True)
    torch.manual_seed(7)
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(x @ x.T, (x.float() @ x.float().T).bfloat16(), atol=0.25, rtol=0.01)
    gate, up = x.float().chunk(2, dim=-1)
    torch.testing.assert_close(silu_and_mul(x), (F.silu(gate) * up).bfloat16(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(gelu_tanh_and_mul(x), (F.gelu(gate, approximate="tanh") * up).bfloat16(), atol=0.02, rtol=0.02)
    w = torch.ones(256, device="cuda", dtype=x.dtype)
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(rmsnorm(x, w), expected.bfloat16(), atol=0.02, rtol=0.02)
    pinned = alloc_pinned_tensor(8, 256, dtype=x.dtype)
    pinned.copy_(x.cpu())
    torch.testing.assert_close(pinned.to("cuda"), x)
    indices = torch.tensor([7, 0, 3], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(indexing(x, indices), x[indices.long()])
    k_cache, v_cache = torch.zeros_like(x), torch.zeros_like(x)
    store_cache(k_cache, v_cache, indices, x[:3], x[3:6])
    torch.testing.assert_close(k_cache[indices.long()], x[:3])
    torch.testing.assert_close(v_cache[indices.long()], x[3:6])
    output = torch.empty((3, 256), device="cuda", dtype=x.dtype)
    dst_indices = torch.arange(3, device="cuda", dtype=torch.int32)
    fast_index_copy_jit(output, dst_indices, pinned, indices)
    torch.testing.assert_close(output, x[indices.long()])
    key = torch.tensor([1, 2, 3], dtype=torch.int64)
    assert fast_compare_key(key, key.clone()) == 3
    torch.cuda.synchronize()
    print("PASS: BF16 matmul, activations, RMSNorm, pinned memory, index/store/offload JIT, radix", flush=True)


if __name__ == "__main__":
    main()
