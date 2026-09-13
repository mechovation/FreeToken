"""Compare full-layer and resident-reuse copies at Qwen3.6 NVFP4 row sizes."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from freetoken.moe.offload_cache import OffloadMoeCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(42)
    e = 256
    cache = OffloadMoeCache(num_layers=1, num_experts=e, cache_size=3 * e,
                           device=torch.device("cuda"), prefill_overlap=True,
                           prefill_hit_d2d=True, quant_format="nvfp4")
    sizes = (1048576, 131072, 2048, 524288, 65536, 4096)
    sources = {name: [torch.randint(0, 256, (e, size), dtype=torch.uint8).pin_memory()]
               for name, size in zip(cache.bank_schema, sizes)}
    cache.set_bank_sources(sources)
    cache.begin_prefill()
    assert cache._prefill_hit_d2d_active
    native = cache._batch_memcpy

    def tensor_copies(dsts, srcs, lengths, stream):
        # Reference submission via PyTorch slices; called on the copy stream.
        for dst, src, length in zip(dsts.tolist(), srcs.tolist(), lengths.tolist()):
            for per_layer, bank in cache.banks:
                base = per_layer[0].data_ptr()
                if base <= src < base + per_layer[0].numel():
                    bank.view(-1)[dst - bank.data_ptr():dst - bank.data_ptr() + length].copy_(
                        per_layer[0].view(-1)[src - base:src - base + length], non_blocking=True)
                    break
            else:
                raise AssertionError("unknown source pointer")

    def run():
        cache.begin_prefill()
        cache.prefetch_prefill_layer(0)
        views = cache.wait_prefill_layer(0)
        cache.release_prefill_layer(0)
        torch.cuda.synchronize()
        return views

    results = []
    for pattern in ("contiguous", "fragmented"):
        order = torch.arange(e) if pattern == "contiguous" else torch.randperm(e)
        for nhit in (0, 64, 192, 256):
            hits = order[:nhit].to("cuda")
            cache.slot_for_id.fill_(-1)
            cache.id_of_slot.fill_(-1)
            cache.slot_for_id[0, hits] = torch.arange(2 * e, 2 * e + nhit, device="cuda", dtype=torch.int32)
            cache.id_of_slot[2 * e:2 * e + nhit] = hits.to(torch.int32)
            for per_layer, bank in cache.banks:
                bank[2 * e:2 * e + nhit].copy_(per_layer[0][order[:nhit]])
            for modes in (("full", "tensor", "hip_batch"), ("hip_batch", "tensor", "full")):
                for mode in modes:
                    cache.prefill_hit_d2d = mode != "full"
                    cache._batch_memcpy = tensor_copies if mode == "tensor" else native
                    for _ in range(2):
                        views = run()
                    for view, per_layer in zip(views, sources.values()):
                        assert torch.equal(view.cpu(), per_layer[0])
                    samples = []
                    for _ in range(10):
                        start = time.perf_counter()
                        run()
                        samples.append(1000 * (time.perf_counter() - start))
                    row = {"pattern": pattern, "hits": nhit, "backend": mode,
                           "median_ms": statistics.median(samples), "samples_ms": samples}
                    results.append(row)
                    print(pattern, nhit, mode, round(row["median_ms"], 3), flush=True)
    a.output.write_text(json.dumps({"torch": torch.__version__, "row_bytes": sizes,
                                   "experts": e, "timings": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
