# HIP graphs for FreeToken on the R9700

Status: completed September 13, 2026. Capture/replay, numerical, serving,
rebuild, controlled A/B benchmarks and deployment checks passed. Prior services
were restored. See [results](rocm-hip-graphs-results.md) and
[session handoff](rocm-hip-graphs-handoff.md).

Implement and validate decode graph capture/replay for
`nvidia/Qwen3.6-35B-A3B-NVFP4` on the Radeon AI PRO R9700 (`gfx1201`, 32 GB).
Use the existing shared PyTorch graph runner. Begin by testing it unchanged;
make code changes only for demonstrated capture, replay, or lifecycle defects.

## Starting evidence and scope

- [GraphRunner](../python/freetoken/engine/graph.py) already creates
  `torch.cuda.CUDAGraph` objects, captures model decode, updates persistent input
  buffers, and replays graphs. There is no blanket HIP exclusion here.
- [Triton attention](../python/freetoken/attention/triton.py) already provides
  capture buffers and metadata updates for replay. Qwen's recurrent state uses
  persistent slot indices through `GraphCaptureBuffer` and
  [FLAMetadata](../python/freetoken/attention/linear.py).
- [Compose](../compose.yaml) passes `ROCM_GRAPH_MAX_BS`, defaulting to zero.
  The [ROCm Dockerfile](../docker/Dockerfile.rocm) also disables capture in its
  default command. This was an initial validation choice; the saved reports
  do not establish a failed HIP graph experiment.
- Existing tests check attention metadata and mock graph warmup ordering, but
  those checks do not establish correct real HIP capture and repeated replay.
- The [benchmark report](rtx3060-benchmark-comparison.md) measured NVIDIA decode
  at 22.18 tok/s without graphs and 42.90 with graphs at the same 64K KV floor.
  AMD's graph-disabled baseline was 19.47 tok/s despite capacity for all 10240
  experts. Neither the NVIDIA speedup nor AMD's cache capacity predicts the
  result of an actual AMD graph test.

HIP provides graph capture/replay, and AMD documents PyTorch's
`torch.cuda.CUDAGraph` support. The CUDA names can remain in shared Python code;
a separate native HIP graph engine is not the starting requirement.
[AMD PyTorch compatibility](https://rocm.docs.amd.com/en/docs-6.4.0/compatibility/ml-compatibility/pytorch-compatibility.html)
documents support predating the installed ROCm 7.2 stack.

Keep the current PyTorch `2.11.0+rocm7.2`, Triton 3.6.0, portable NVFP4 kernels,
and pure-PyTorch router for the initial graph comparison. ROCm 10 migration,
router fusion, native FP8/FP4 optimization, prefill graphs, CPU/hybrid experts,
other model families, and multi-GPU support are separate follow-ups.

## 1. Reproduce the baseline and try capture

Record the source snapshot, image digest, driver/runtime versions, checkpoint
revision, startup arguments, and actual cache geometry. Preserve the existing
working-tree edits and cached image. Use a separate test image/tag and Compose
override for any fixes; retain the current working deployment for rollback.

During implementation, record service state and quiesce competing inference
before tests. Serialize the two GPUs' 35B model loads if needed: they share
roughly 64 GB of host RAM, and earlier NVIDIA startup caused swapping. Restore
the prior services when experiments finish; do not benchmark through active
memory pressure or another model's workload.

First run a small HIP graph smoke test in the existing ROCm image, without
model weights: warmed BF16 arithmetic, a representative Triton operation, and
the native mapped-host expert-copy kernel. Modify inputs between at least 100
replays and compare against independently computed references. Compile JIT
variants and initialize libraries before capture; synchronize outside capture
when checking results.

Then attempt the full NVFP4 model unchanged with graph maximum batch size 1,
followed by 2. Keep maximum active requests at 2 and the baseline's KV floor at
262144, memory ratio 0.85, Triton attention, expert offload, and prefill chunk
2048. Use the same cached model revision and serial expert loading.

Decision: if capture and replay already work, proceed to correctness testing
and integration. If they fail, save the exact error, failing operation, batch
size, and software versions, then reduce it to a small reproduction. A simple
runtime smoke test passing does not establish full-model support; one failed
operation does not establish that HIP graphs are generally unsupported.

## 2. Fix demonstrated capture and replay defects

HIP capture has stream/synchronization restrictions, and Python-side work is
not automatically rerun by replay. Follow the documented capture rules rather
than suppressing errors or changing capture mode to conceal them.
[HIP capture rules](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html#capture-graphs-from-a-stream)
and [PyTorch graph constraints](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)
provide the reference behavior.

| Area | Files to inspect/change if required | Required behavior |
|---|---|---|
| Capture preparation | `python/freetoken/engine/graph.py`, `engine/engine.py` | Warm required kernels before capture; use the correct stream; preserve buffer addresses; initialize dummy state; synchronize outside capture. |
| Attention metadata | `python/freetoken/attention/triton.py` | Update positions, sequence lengths, page indices, and scratch data for every replay without rebinding captured storage. |
| Qwen recurrent state | `python/freetoken/attention/linear.py`, `kernel/fla/`, `kvcache/linear_state_pool.py`, `models/qwen3_5_moe/gdn.py`, `models/qwen3_5_moe/gdn_kernels.py` | Update the correct request's convolution/GatedDeltaNet state, preserve dummy-slot isolation, and survive slot reuse. |
| Expert routing and cache | `python/freetoken/layers/moe.py`, `moe/offload_cache.py`, `moe/offload_kernels.py`, `moe/fused.py` | Dynamic expert IDs and miss counts must drive captured GPU work; changing routes must not reuse capture-time results. |
| Native expert copies | `python/freetoken/kernel/fast_index_copy.py`, `kernel/csrc/jit/fast_index_copy.cuh`, `kernel/csrc/include/freetoken/hip_compat.cuh` | Launch on the capturing/replaying stream and use valid device-visible pinned-bank pointers. Exercise actual misses, not only warm cache hits. |
| Teardown/rebuild | `python/freetoken/engine/engine.py`, `engine/graph.py`, `attention/triton.py` | Destroy graphs before freeing their storage; rebuild buffers and recapture before reuse. |

Keep the pure-PyTorch router initially. Its sequence of operations may be
capturable without replacing it; assess fusion later with a separate A/B.
Keep prefill eager, including its copy-stream/event ordering. Test the handoff
between eager prefill and captured decode. Do not enable the separate
NVIDIA-specific batched prefill-copy optimization as part of this work.

Use shared fixes where appropriate and narrow HIP branches where necessary.
Retain existing CLI names and CUDA behavior. Avoid a broad naming refactor or
new graph abstraction unless a concrete runtime incompatibility requires one.

## 3. Prove correctness and lifecycle safety

Add real GPU graph tests alongside the existing numerical tests. Compare
kernel results to independent PyTorch/dequantization references and graph
execution to eager execution from equivalent initial state. Use existing
dtype-appropriate tolerances; investigate mismatches before relaxing them.
Reuse the model's `python/freetoken/models/qwen3_5_moe/gdn_reference.py` where
applicable. Restore equivalent active KV pages and recurrent state between
comparisons; avoid allocating two full 35B model instances simultaneously.

| Test level | Coverage and acceptance |
|---|---|
| Runtime/kernel | At least 100 replays with changing inputs; batch sizes 1 and 2; finite, correct outputs and stable live buffer addresses. |
| Attention/state | Vary positions, KV page mappings, sequence lengths, and recurrent-state slots; compare outputs and state updates to references; no cross-request contamination. |
| Expert cache | Cold cache after capture, warm hits, forced misses/evictions in a small synthetic cache, then full-capacity model cache. Verify copied data, selected experts, and outputs. |
| Full-model math | Teacher-force the same token sequence for at least 32 decode steps at batch sizes 1 and 2; compare logits and relevant KV/GDN state with eager runs. Avoid relying only on sampled text equality. |
| Serving | Coherence and fixed greedy cases; concurrent requests entering/leaving; transitions 1→2→1; prefix reuse; request cancellation and slot reuse; at least 100 completed short requests without hangs, errors, or memory growth. |
| Lifecycle | Repeated start/stop, supported cache resize/recapture, invalid resize rejection, and continued correct serving after rebuild. |
| Context | Successful 16K and near-64K prompts plus decode, using the same lengths on comparison runs. Validate larger reservations separately from filled input length. |

Extend `tests/kernels/test_triton_attention.py`, `test_pinned_tensor.py`,
`tests/moe/test_nvfp4_backends.py`, and `tests/e2e/test_cache_rebuild.py` where
their references fit. Add a focused `tests/engine/test_graph_replay.py` and
`scripts/rocm-graph-smoke.py` for real replay coverage. New files are proposed,
not already implemented. Keep hardware tests explicit about their requirements;
a skipped HIP test does not count as validation. Run affected shared tests and
a CUDA capture/serving regression on the 3060 when shared production code changes.

## 4. Make effective graph use observable

Startup diagnostics should report runtime backend, requested graph sizes,
successfully captured sizes, capture duration, and added GPU memory. Preserve
`/health` readiness until initialization has completed.

Expose or collect aggregate decode replay counts and eager-decode counts with
reasons. Prefill is expected to remain eager and must be counted separately.
Use a short diagnostic trace to verify actual HIP graph launches and remaining
CPU/metadata overhead. Capture success or a positive CLI flag alone is not
proof that serving requests use graph replay. Keep tracing and synchronization
out of the throughput measurements.

An explicitly requested capture failure must remain visible and fail startup
with useful context. Do not silently report graph-enabled performance after
falling back to eager mode. If capture invalidates the stream/context, exit and
restart cleanly rather than attempting to reuse questionable state.

## 5. Measure the AMD graph benefit and expert residency

Use the same candidate source/image for fresh AMD graphs-off and graphs-on
runs. First compare at the historical 262144-token shared KV floor while
retaining capacity for all 10240 experts. Then repeat with a 65536-token floor
for the user's preferred configuration. Record actual allocations in every run:
the floor can leave AMD with more KV tokens than requested after all experts fit.

Hold actual expert/KV geometry constant within each AMD A/B pair; if automatic
sizing varies, pin the observed safe sizes for both runs and report the change.
For an additional comparison with equal actual 64K KV allocation across GPUs,
use the existing `--num-tokens 65536` override with a matching KV floor and
verify `/v1/cache/status`. Keep AMD's expert-capacity advantage. Do not combine
fixed expert sizing with the mutually exclusive `--moe-cache-auto` flag.

Use `llama-benchy==0.4.0`, the cached identical tokenizer/corpus, 512 prompt
tokens, output budget 128, depth zero, concurrency 1 and 2, thinking off, no
client caching or prompt adaptation, and five measured runs after warmup.
Repeat the paired comparison in reverse order. Retain per-run results and
report variability; historical NVIDIA numbers remain labeled historical unless
rerun under the same controls.

Separately measure single-request 16384- and 65000-token prompts with an output
budget of 64, including end-to-end first-token latency. Do not present a KV
reservation as an actual input length, or compare unlike prompt lengths as a
hardware A/B. Preserve reported token counts and document the existing 127/128
and 63/64 accounting discrepancies instead of silently normalizing results.

In a separate diagnostic run, use existing MoE cache statistics and a profiler
to measure actual expert misses/copies, CPU dispatch time, and GPU kernel time.
Initialize statistics before capture and reset warmup counts before sampling;
read them outside captured execution. Verify how the installed runtime exposes
the existing counters before adding another stats interface. The R9700's
10240-slot capacity must not be assumed to imply zero transfers.

Save raw JSON, server/client logs, graph-use evidence, source/image identity,
cache geometry, memory/swap observations, and reproduction commands in a new
HIP-graph result directory. Keep graph profiling separate from timed runs.

## 6. Enable the validated deployment and retain rollback

Once correctness, real replay, and stability pass, enable graph batch sizes
1 and 2 for the tested R9700/Qwen configuration through `ROCM_GRAPH_MAX_BS=2`
and `ROCM_MAX_REQUESTS=2`. Update `compose.yaml`, `.env.rocm.example`, and
`docs/docker-rocm.md` consistently for that deployment. Review the Dockerfile's
separate small-model command independently rather than claiming all models
are validated. Keep `ROCM_GRAPH_MAX_BS=0` as the explicit opt-out.

Choose the 64K KV floor for the final user configuration and report its actual
allocation. Record any additional memory required by capture. Do not silently
reduce the expert cache to obtain a speedup or fit a graph; if a reduction is
required, measure and disclose it as a separate configuration.

Update `docs/rocm-validation-summary.md` and the NVIDIA comparison with the
new graph-enabled AMD results and clearly dated service state. Retain the
previous image and a tested graph-disabled launch for rollback. Restore
unrelated services to their recorded states after benchmarking.

Completion means: demonstrated HIP capture and real replay, passing numerical
and serving tests, stable memory/lifecycle behavior, preserved CUDA behavior,
reproducible AMD A/B results, and an explicitly configured working deployment.
There is no predetermined speedup target. If correct graph replay gives little
benefit, use the trace to prioritize remaining router, conversion, transfer,
or kernel costs in a separate optimization change.
