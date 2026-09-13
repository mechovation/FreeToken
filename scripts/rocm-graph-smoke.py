"""Real HIP graph replay checks; run in the ROCm image without model weights."""

import torch
import torch.nn.functional as F

from freetoken.kernel.fast_index_copy import fast_index_copy_jit
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.kernel.triton.activation import silu_and_mul


def main():
    assert torch.version.hip, "Requires ROCm PyTorch"
    torch.manual_seed(7)
    print(f"torch={torch.__version__}, HIP={torch.version.hip}", flush=True)
    print(torch.cuda.get_device_properties(0), flush=True)
    for bs in (1, 2):
        x = torch.empty(bs, 256, device="cuda", dtype=torch.bfloat16)
        host = alloc_pinned_tensor(8, 256, dtype=x.dtype)
        host.copy_(torch.arange(2048).reshape(8, 256).to(x.dtype))
        src = torch.zeros(bs, device="cuda", dtype=torch.int32)
        dst = torch.arange(bs, device="cuda", dtype=torch.int32)
        count = torch.zeros(1, device="cuda", dtype=torch.int64)
        copied = torch.zeros_like(x)

        def forward():
            fast_index_copy_jit(copied, dst, host, src, count)
            return x @ x.T, silu_and_mul(x)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x.zero_()
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            matmul, activated = forward()
        pointers = tuple(t.data_ptr() for t in (x, src, copied, matmul, activated))
        # Capture used zero misses. Replays must still perform changing native
        # host-bank copies, including partial/no-copy cases and changing routes.
        for step in range(100):
            cpu_x = torch.randn(bs, 256, dtype=x.dtype)
            ids = (torch.arange(bs) + step) % 8
            misses = step % (bs + 1)
            x.copy_(cpu_x)
            src.copy_(ids)
            count.fill_(misses)
            copied.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.full((bs, 256), -1, dtype=x.dtype)
            expected[:misses] = host[ids[:misses]]
            torch.testing.assert_close(copied.cpu(), expected, rtol=0, atol=0)
            ref = (cpu_x.float() @ cpu_x.float().T).to(x.dtype)
            torch.testing.assert_close(matmul.cpu(), ref, rtol=0.01, atol=0.25)
            gate, up = cpu_x.float().chunk(2, dim=-1)
            ref = (F.silu(gate) * up).to(x.dtype)
            torch.testing.assert_close(activated.cpu(), ref, rtol=0.02, atol=0.02)
            assert pointers == tuple(
                t.data_ptr() for t in (x, src, copied, matmul, activated)
            )
        print(
            f"PASS: bs={bs}, 100 changing HIP replays, BF16/Triton/native host copies",
            flush=True,
        )


if __name__ == "__main__":
    main()
