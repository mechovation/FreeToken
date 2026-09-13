# ROCm prefill expert reuse exploration

September 13, 2026. Approved exploration; see
[results](rocm-prefill-reuse-results.md) for implementation and measurements.
Keep the PyTorch router and validated HIP graphs at batches 1/2.

## Objective

Determine whether reusing GPU-resident experts improves Qwen3.6-35B-A3B NVFP4
prefill latency on AMD, without degrading decode, cache correctness, or serving
reliability. Deliver a measured recommendation; enabling the existing optional
path is not itself a success criterion.

## Starting point

- `moe/offload_cache.py` already implements `prefill_hit_d2d`, enabled by
  `--moe-prefill-hit-d2d`, with a full-layer-copy fallback.
- It requires prefill overlap, the fused copy plan, and more than `2E` cache
  slots, where `E` is experts per layer. The first `2E` slots are volatile
  double buffers; they cannot be treated as reusable resident hits.
- `begin_prefill()` snapshots the slot map to pinned CPU memory and synchronizes
  the copy stream. `_prefetch_split()` gathers hits on the compute stream,
  coalesces miss runs on the host, and submits copies on the prefill stream.
- Small banks are copied in full, including when all large-bank rows hit.
  Therefore expert hit rate alone does not quantify bytes saved.
- `kernel/batch_memcpy.py` explicitly requires CUDA 13 and rejects ROCm. This
  is a known binding limitation, not evidence that reuse cannot work on AMD.
- Existing tests in `tests/moe/test_prefill_hit_d2d.py` cover basic mixtures,
  poisoned volatile slots, all-hit/all-miss cases, and insufficient capacity.
  Most are gated on CUDA 13 and currently cannot validate the AMD path.

## 1. Establish the baseline and opportunity

Record exact image/source IDs, effective settings, bank layouts and byte sizes,
actual KV capacity, expert capacity, and whether overlap is active. Start with
the handoff's 65536 actual KV-token configuration, then test normal auto-sizing.
Compare configurations at the same actual capacities rather than just the same
KV floor. Confirm the router remains on the PyTorch fallback.

Collect untimed diagnostics for cold cache, decode-warmed cache, and consecutive
prefill chunks. Measure reusable rows outside the volatile region, H2D bytes by
bank, number and size of miss runs, copy duration, compute waiting for copies,
and cache residency before/after prefill. Separate expert-cache warmth from KV
prefix-cache hits: use fresh prompts for the primary comparison.

Measure the existing full-layer copies and estimate an optimistic saving from
eliminating reusable large-bank transfers. Account for work already hidden by
overlap. If exposed copy time or reusable residency is negligible, document that
and stop before implementing a port.

Deliverable: baseline artifacts and a breakdown showing where reuse could help.

## 2. Test an AMD-compatible copy path in isolation

Inspect the installed HIP headers/runtime for usable copy APIs and verify their
semantics with real copies; do not assume a CUDA batch API has a drop-in HIP
equivalent. Compare these bounded alternatives using production bank sizes:

1. Coalesced nonblocking tensor-slice H2D copies on the existing copy stream:
   a simple reference candidate using the installed PyTorch ROCm stack.
2. A small native wrapper submitting HIP asynchronous copies per coalesced run,
   if Python submission overhead makes the reference candidate inadequate.
3. A native HIP batch API only if the installed stack supports the required
   semantics and measurement justifies it.

Reuse the existing hit compaction and device gather after validating them on
AMD. Retain the stream/event protocol, volatile-slot exclusion, and small-bank
handling. Keep CUDA dispatch intact and report the selected backend or fallback
reason. Unsupported configurations must retain the working full-layer path.

Benchmark zero/partial/full hits and both contiguous and fragmented misses.
Include snapshot synchronization, host run construction, D2D gather, H2D work,
and compute interference in the cost. Raw transfer bandwidth alone is not the
decision metric. Consider an adaptive cutoff only if repeatable data supports
one; start with an explicit opt-in experiment.

Deliverable: the smallest functioning candidate plus a microbenchmark against
full-layer copies. Stop if it cannot beat the baseline in relevant warm cases.

## 3. Validate cache and stream correctness

Extend existing tests rather than replacing their CUDA coverage. Require the
AMD tests to assert that reuse actually activates, so fallback cannot produce a
false pass. Compare every materialized bank byte-for-byte with host sources.

Cover all-hit/all-miss/mixed and fragmented patterns; poisoned slots below
`2E`; exact and insufficient capacity thresholds; NVFP4 data, scale and global
banks; multiple layers and repeated buffer wraps; and multiple prefill chunks.
Include deliberately delayed copy/compute work to exercise release/ready fences.

Verify that snapshot and live slot maps agree on reusable residency for the
whole chunk. Audit scheduler/cache writers rather than assuming decode cannot
modify the hit region concurrently. Exercise decode → prefill → decode,
evictions, cache rebuilds across capacity thresholds, and fallback paths.

Prefill itself need not become graph-captured. Existing HIP decode graphs must
continue to replay correctly after prefill and must be rebuilt safely when
cache storage changes. Check shared changes on the RTX 3060 as well.

Deliverable: passing bank/cache tests and unchanged graph correctness checks.

## 4. Measure full-model behavior

Use isolated candidate and baseline runs with the same model, capacity, router,
graph sizes, overlap settings, and prompt/output lengths. Serialize 35B loads
across GPUs because they share host RAM. Before AMD model runs, inspect current
services, unload the active llama-swap model, and restore its prior state when
finished. Do not run both AMD models together.

Test cold and deliberately warmed expert caches with fresh prompts of roughly
512, 2048, 8192 and 16384 tokens; extend to about 65000 only if initial results
justify the longer run. Include repeated chunks, prefix reuse as a separate
scenario, and concurrent requests at the configured maximum of two.

Report TTFT, prefill tokens/s, decode tokens/s, latency variability, H2D/D2D
bytes, hit rate, actual memory/capacity and host swapping. Collect diagnostic
counters separately from throughput runs. Use at least ten samples for each
primary short comparison, reverse A/B ordering, and reset or recreate cache
state consistently. Record actual processed tokens for prefix-cache scenarios.

Compare model outputs/logits with reuse off/on from matched input and KV/GDN
state, using established tolerances without relaxing them. Exercise repeated
serving, cancellation, prefix reuse and rebuild/recapture before recommending
deployment. Add a long-context confirmation if shorter workloads improve.

## 5. Decision and handoff

Recommend enabling reuse only when correctness passes and repeated full-model
measurements show a warm-prefill TTFT improvement beyond observed run-to-run
variation, without a material cold-prefill, decode, or memory regression.
Report both absolute and relative changes; a copy microbenchmark win is
insufficient. If the benefit is workload-dependent, retain opt-in use and state
the useful conditions. If it loses, preserve the fallback and document why.

Save raw measurements under `benchmarks/results/rocm-prefill-reuse/` and write
`docs/rocm-prefill-reuse-results.md` with the tested configuration, limitations,
reproduction commands, recommendation and rollback. Update the HIP handoff.
Keep production defaults unchanged during exploration.
