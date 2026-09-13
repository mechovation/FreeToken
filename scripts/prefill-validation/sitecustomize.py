"""Opt-in prefill diagnostics. Never use this probe for throughput measurements."""

import json
import os
import time

if os.getenv("FREETOKEN_PREFILL_DIAGNOSTICS") == "1":
    import torch
    from freetoken.moe.offload_cache import OffloadMoeCache, _SMALL_BANK_FEAT_BYTES

    _begin = OffloadMoeCache.begin_prefill
    _prefetch = OffloadMoeCache.prefetch_prefill_layer
    _wait = OffloadMoeCache.wait_prefill_layer
    _release = OffloadMoeCache.release_prefill_layer

    def event():
        return torch.cuda.Event(enable_timing=True)

    def begin(self):
        _begin(self)
        if not self.prefill_overlap:
            print("PREFILL_PROBE " + json.dumps({"overlap": False}), flush=True)
            return
        # Diagnostic synchronization deliberately removes preceding decode work.
        torch.cuda.synchronize()
        self._probe_snapshot = self.slot_for_id.cpu()
        self._probe_layers = {}
        self._probe_chunk = getattr(self, "_probe_chunk", 0) + 1

    def prefetch(self, layer):
        if layer >= self.num_layers or self._prefill_buffer_layer[layer % 2] == layer:
            return _prefetch(self, layer)
        start, end = event(), event()
        host_start = time.perf_counter()
        start.record(self.prefill_copy_stream)
        _prefetch(self, layer)
        end.record(self.prefill_copy_stream)
        self._probe_layers[layer] = {"copy_events": (start, end),
                                     "host_submit_ms": 1000 * (time.perf_counter() - host_start)}

    def wait(self, layer):
        start, end = event(), event()
        start.record()
        result = _wait(self, layer)
        end.record()
        self._probe_layers[layer]["wait_events"] = (start, end)
        return result

    def release(self, layer):
        _release(self, layer)
        if layer != self.num_layers - 1:
            return
        torch.cuda.synchronize()
        feats = [bank[0].numel() * bank.element_size() for _, bank in self.banks]
        rows = []
        for i, times in self._probe_layers.items():
            hits = self._probe_snapshot[i] >= 2 * self.num_experts
            miss = (~hits).nonzero().flatten()
            runs = int(miss.numel() > 0) + int((miss.diff() != 1).sum())
            row = {"layer": i, "reusable_rows": int(hits.sum()),
                   "miss_runs": runs,
                   "full_h2d_bytes": sum(feats) * self.num_experts,
                   "avoidable_h2d_bytes": sum(f for f in feats if f >= _SMALL_BANK_FEAT_BYTES) * int(hits.sum())}
            for name, value in times.items():
                if name.endswith("_events"):
                    start, end = value
                    row[name.replace("_events", "_ms")] = start.elapsed_time(end)
                else:
                    row[name] = value
            rows.append(row)
        print("PREFILL_PROBE " + json.dumps({"chunk": self._probe_chunk,
              "overlap": True, "reuse_active": self._prefill_hit_d2d_active,
              "cache_size": self.cache_size, "experts_per_layer": self.num_experts,
              "bank_row_bytes": feats, "layers": rows}), flush=True)

    OffloadMoeCache.begin_prefill = begin
    OffloadMoeCache.prefetch_prefill_layer = prefetch
    OffloadMoeCache.wait_prefill_layer = wait
    OffloadMoeCache.release_prefill_layer = release

if os.getenv("FREETOKEN_PREFILL_VERIFY") == "1":
    import torch
    from freetoken.layers.moe import OffloadMoELayer

    _routed = OffloadMoELayer._prefill_routed

    def routed(self, hidden, weights, ids):
        result = _routed(self, hidden, weights, ids)
        cache = self.offload_cache
        assert cache._prefill_hit_d2d_active, "reuse validation must not silently fall back"
        views = tuple(buffer[self.layer_id % 2] for buffer in cache.prefill_bank_buffers)
        # Same hidden states and routing, but a fresh full-layer copy from host
        # supplies the independent expert-compute reference. No KV/GDN replay.
        for view, (per_layer, _) in zip(views, cache.banks):
            torch.testing.assert_close(view.cpu(), per_layer[self.layer_id], rtol=0, atol=0)
            view.copy_(per_layer[self.layer_id], non_blocking=True)
        expected = self._expert_gemm(cache, hidden, weights, ids, views=views,
                                    n=self.num_experts,
                                    alphas=cache.alphas_for_layer(self.layer_id), is_prefill=True)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
        print("PREFILL_VERIFY " + json.dumps({"layer": self.layer_id,
              "tokens": hidden.shape[0], "bank_bytes": "exact", "expert_output": "pass"}), flush=True)
        return result

    OffloadMoELayer._prefill_routed = routed

if os.getenv("FREETOKEN_VALIDATE_GRAPHS") == "1":
    import runpy

    runpy.run_path("/graphprobe/sitecustomize.py")
