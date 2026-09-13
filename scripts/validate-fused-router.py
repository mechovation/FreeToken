"""Validate the optional router and time it against FreeToken's torch fallback.

Run in an existing GPU environment with triton_kernels on PYTHONPATH. No model
is loaded. Output is JSON; timings include the complete FreeToken router wrapper.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
import triton
import triton_kernels

from freetoken.moe.fused import _torch_fused_topk, fused_topk


def check(logits, weights, ids, k, renormalize, valid):
    # CPU float64 oracle, independently evaluated from the GPU fallback. Ties
    # may choose different experts: require unique IDs and a valid top-k cutoff.
    x = logits.cpu().double()
    w, idx = weights.cpu(), ids.cpu().long()
    assert w.dtype == torch.float32 and ids.dtype == torch.int32
    assert w.shape == idx.shape == (x.shape[0], k)
    assert weights.is_contiguous() and ids.is_contiguous()
    assert (idx[valid:] == -1).all()
    idx = idx[:valid]
    assert ((idx >= 0) & (idx < x.shape[1])).all()
    assert (idx.sort(dim=-1).values.diff(dim=-1) > 0).all()
    selected = x[:valid].gather(1, idx)
    # Without renormalization the wrapper selects *after* FP32 softmax.
    # Extreme logits can underflow to tied zero probabilities, so selection
    # must be judged in that space, not against the original logits.
    scores = x[:valid] if renormalize else logits[:valid].float().softmax(-1).cpu()
    cutoff = scores.topk(k, dim=-1).values[:, -1:]
    assert (scores.gather(1, idx) >= cutoff).all()
    expected = (selected.softmax(-1) if renormalize else
                x[:valid].softmax(-1).gather(1, idx))
    torch.testing.assert_close(w[:valid].double(), expected, atol=2e-7, rtol=2e-5)


def time_graph(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    # Amortize event and replay overhead across 100 complete router calls.
    with torch.cuda.graph(graph):
        for _ in range(100):
            fn()
    samples = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(20260913)
    root = Path(triton_kernels.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    result = {"torch": torch.__version__, "triton": triton.__version__,
              "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(),
              "router_source_sha256": digest.hexdigest(), "cases": 0,
              "graph_replays": 0, "timings": []}
    for batch in (1, 2, 33, 512):
        for experts, k in ((32, 4), (64, 8), (256, 8), (257, 8)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for renorm in (False, True):
                    for pattern in ("random", "ties", "extreme"):
                        x = torch.randn(batch, experts, device="cuda", dtype=dtype)
                        if pattern == "ties":
                            x.zero_()
                        elif pattern == "extreme":
                            x.mul_(100)
                        for valid in (batch, max(0, batch - 1)):
                            n = (None if valid == batch else
                                 torch.tensor(valid, device="cuda", dtype=torch.int32))
                            w, ids = fused_topk(x, x, k, renorm, n)
                            try:
                                check(x, w, ids, k, renorm, valid)
                            except Exception as exc:
                                raise AssertionError((batch, experts, k, str(dtype), renorm, pattern, valid)) from exc
                            result["cases"] += 1
    for batch in (1, 2):
        for renorm in (False, True):
            x = torch.randn(batch, 256, device="cuda")
            n = torch.tensor(batch, device="cuda", dtype=torch.int32)
            for _ in range(3):
                fused_topk(x, x, 8, renorm, n)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                w, ids = fused_topk(x, x, 8, renorm, n)
            for step in range(args.replays):
                x.normal_()
                valid = step % (batch + 1)
                n.fill_(valid)
                graph.replay()
                check(x, w, ids, 8, renorm, valid)
                result["graph_replays"] += 1
    for batch in (1, 2, 32, 512):
        x = torch.randn(batch, 256, device="cuda", dtype=torch.bfloat16)
        for renorm in (False, True):
            fns = {"torch": lambda: _torch_fused_topk(x, 8, renorm, None),
                   "fused": lambda: fused_topk(x, x, 8, renorm)}
            # Reverse order on the second pass to expose ordering effects.
            for order in (tuple(fns), tuple(reversed(fns))):
                for name in order:
                    result["timings"].append({"batch": batch, "renormalize": renorm,
                                              "backend": name,
                                              **time_graph(fns[name], args.repeats)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
