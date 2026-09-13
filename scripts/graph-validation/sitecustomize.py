"""Opt-in full-model graph/eager probe, loaded via PYTHONPATH in test containers.

Never enable during throughput benchmarks: this intentionally synchronizes and
runs decode twice. Same input tokens and initial KV/GDN state feed both paths.
Only newly written KV rows need restoration; all GDN slots are checked to catch
cross-request writes. The normal graph result/state is retained for serving.
"""

import json
import os

if (
    os.environ.get("FREETOKEN_VALIDATE_GRAPHS") == "1"
    or os.environ.get("FREETOKEN_GRAPH_STATS_ONLY") == "1"
):
    import torch
    from freetoken.core import get_global_ctx
    from freetoken.engine.engine import Engine
    from freetoken.engine.graph import GraphRunner

    _init = Engine.__init__
    _replay = GraphRunner.replay

    def init(self, config):
        object.__setattr__(config, "moe_collect_stats", True)
        _init(self, config)
        self.graph_runner._probe_model = self.model
        self.graph_runner._probe_steps = {}

    def replay(self, batch):
        bs = batch.size
        steps = getattr(self, "_probe_steps", {})
        if os.environ.get("FREETOKEN_GRAPH_STATS_ONLY") == "1":
            n = steps.get(bs, 0)
            if n in (0, 32, 64, 128, 256):
                self.moe_offload_cache.reset_stats()
            result = _replay(self, batch)
            steps[bs] = n + 1
            self._probe_steps = steps
            if n + 1 in (32, 64, 128, 256, 512):
                stats = self.moe_offload_cache.decode_miss_stats()
                # fetched/cpu columns of the existing API are hybrid-only; the
                # plain offload path copies every missing expert to the GPU.
                record = {
                    "batch_size": bs,
                    "through_decode_step": n + 1,
                    "memory_allocated_bytes": torch.cuda.memory_allocated(),
                    "memory_reserved_bytes": torch.cuda.memory_reserved(),
                    **{
                        k: stats[k]
                        for k in (
                            "layer_calls",
                            "active_per_layer",
                            "missing_per_layer",
                            "miss_rate",
                        )
                    },
                }
                print("GRAPH_EXPERT_STATS " + json.dumps(record), flush=True)
            return result
        if steps.get(bs, 0) >= 32 or not hasattr(self, "_probe_model"):
            return _replay(self, batch)
        ctx = get_global_ctx()
        assert ctx.page_size == 1
        kv = ctx.kv_cache._kv_buffer
        loc = batch.out_loc.long()
        linear = ctx.linear_state_pool
        conv, recurrent = linear.conv_states, linear.recurrent_states
        before = (kv[:, :, loc].clone(), conv.clone(), recurrent.clone())
        if steps.get(bs, 0) == 31:
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                actual = _replay(self, batch).clone()
                torch.cuda.synchronize()
            prefix = os.environ.get("FREETOKEN_GRAPH_TRACE_PREFIX", "hip-graph")
            prof.export_chrome_trace(f"/results/{prefix}-bs{bs}-trace.json")
        else:
            actual = _replay(self, batch).clone()
        after = (kv[:, :, loc].clone(), conv.clone(), recurrent.clone())
        kv[:, :, loc] = before[0]
        conv.copy_(before[1])
        recurrent.copy_(before[2])
        eager = self._probe_model.forward().float()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, eager[:bs], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(kv[:, :, loc], after[0], rtol=1e-3, atol=1e-3)
        # assert_close needs several tensor-sized temporaries. Comparing the
        # full recurrent pool at auto-sized KV capacity can exhaust VRAM even
        # though serving fits. Check every layer with unchanged tolerances.
        for state, expected in ((conv, after[1]), (recurrent, after[2])):
            for layer in range(state.shape[0]):
                torch.testing.assert_close(state[layer], expected[layer], rtol=1e-3, atol=1e-3)
        kv[:, :, loc] = after[0]
        conv.copy_(after[1])
        recurrent.copy_(after[2])
        steps[bs] = steps.get(bs, 0) + 1
        self._probe_steps = steps
        if steps[bs] in (1, 32):
            print(
                "GRAPH_VALIDATION "
                + json.dumps(
                    {
                        "batch_size": bs,
                        "steps": steps[bs],
                        "logits_max_abs": (actual - eager[:bs]).abs().max().item(),
                        "kv_gdn_state": "pass",
                        "moe": self.moe_offload_cache.decode_miss_stats(),
                    }
                ),
                flush=True,
            )
        return actual

    Engine.__init__ = init
    GraphRunner.replay = replay
