# HIP graph implementation handoff

Completed September 13, 2026. No benchmarks or diagnostic processes remain
running. This is a good checkpoint for a fresh session before further tuning.

## Outcome

HIP graphs already worked through FreeToken's shared PyTorch graph runner.
The unchanged cached AMD image captured batch size 1 and served coherently.
The implemented change enables validated graph sizes 1/2 by default for
Qwen3.6 NVFP4, adds HIP-specific diagnostics and actual replay/eager counters,
and provides real numerical, serving, lifecycle and benchmark evidence.

At **65536 actual shared KV tokens**, with all 10240 experts fitting:

| AMD workload | Graphs off | Graphs on |
|---|---:|---:|
| One request | 20.63 tok/s | 74.67 tok/s |
| Two requests, aggregate | 39.93 tok/s | 106.94 tok/s |

Ten measured samples per concurrency/configuration, including reverse-order
repeats. At 262144 actual KV tokens the corresponding rates were
20.79/73.95 single and 40.07/106.02 aggregate. Filled prompts at 16384/65000
passed with graphs: 67.47/52.99 decode tok/s, 8.16/36.10 seconds TTFT.
The saved September 12 RTX 3060 short rates were 42.90/42.63 tok/s.

Read [full results](rocm-hip-graphs-results.md) and the raw
[artifacts](../benchmarks/results/rocm-hip-graphs/) for controls and limits.

## Validation completed

- 100 changing HIP graph replays at each batch size for arithmetic, Triton,
  mapped-host expert copies, including changing device-side miss counts.
- GraphRunner/CPU oracle, attention/page-mapping oracle, NVFP4 dequantization
  oracle with cold cache/hits/evictions, GDN mathematical recurrence oracle.
- Full model: 32 graph/eager comparisons at each batch size, same input tokens
  and initial KV/GDN state. Correctness tolerances were not relaxed.
- 103 short serving requests, three successful rebuilds/recaptures, invalid
  resize rejection, cancellation recovery, and 4480-token prefix reuse.
- CUDA: shared graph tests, GDN recurrence tests, and the real small-model
  cache-rebuild serving test passed on the 3060 with changed production files.
- Eight AMD benchmark boots, 80 measured short batches and four long samples.
  No swap-out during the before/after intervals; minor system-wide swap-in.
- Untimed expert counters: miss rate fell from 27.29% in the first 32
  single-request decode steps to 0.91% in steps 257–512. The subsequent warmed
  two-request window was 0.19%. GPU allocated/reserved memory remained stable.
- Normal Compose defaults reached readiness and served single/concurrent
  requests with all 10240 experts and **340189 actual KV tokens**. A 64K floor
  allows auto-sizing to allocate more; the benchmark caps were explicit.

## Current services and rollback

- `freetoken-rocm`: **stopped after validation**, now configured for
  `freetoken:rocm-hip-graphs`, graph max 2, requests 2, KV floor 65536.
- `freetoken-nvidia`: **stopped at the user's request after validation**;
  retained with its original `freetoken:cuda13` image, 64K floor and graphs 1/2.
- `llama-swap`: restored and healthy on AMD, including its previously loaded
  `qwen3.6:35b` Vulkan model. `llama-swap-nvidia` and other services unchanged.
- Do not load AMD FreeToken alongside the active llama-swap model. Serialize
  35B loads across GPUs too: they share roughly 64 GiB of host RAM.

Candidate image ID:
`e03edc33baca7fdbb5eb1c88b79bc406e373833c494d80a7238b7a4d69101340`.
Original `freetoken:rocm-gfx1201` retained. To roll back, set
`ROCM_RUNTIME_IMAGE=freetoken:rocm-gfx1201 ROCM_GRAPH_MAX_BS=0` when running
Compose. Deployment and rollback commands are in [docker-rocm.md](docker-rocm.md).

## Changes and reproduction

Production changes are limited to `engine/graph.py` diagnostics/counters,
`engine/engine.py` eager counting, Docker/Compose defaults and the JIT cache
namespace. No expert/router/attention algorithm was changed.
`docker/Dockerfile.hip-graphs` builds incrementally from the cached original
image. `docker/Dockerfile.rocm` also supports the updated setup from its normal
build path. No model download or ROCm upgrade was performed.

New tools: `scripts/rocm-graph-smoke.py`, `scripts/benchmark-hip-graphs.py`,
`scripts/graph-validation/sitecustomize.py`. The benchmark driver supports
`--bench`, `--long`, `--validate --stress`, and `--stats --prefix-reuse`.
Never enable the diagnostic probe during throughput measurements.

Early failures were harness/cache issues, preserved in artifacts: stale
native-copy JIT code; frozen-config assignment; float32/BF16 logit comparison;
and the GDN oracle's key-major versus kernel value-major storage. Corrected
without changing numerical tolerances or model algorithms. HIP traces contain
`hipGraphLaunch`, but this profiler setup does not expose every graph kernel.

## Possible next work, only if requested

The user's CUDA/ROCm parity question is audited in the results document:

1. Validate the optional fused `triton_kernels` router on AMD; currently absent,
   with a working separate-operation PyTorch fallback. Do not assume it fails.
2. Measure/port prefill reuse of already-resident experts. Current optional
   batched-copy implementation depends on CUDA; it was off on both GPUs.
3. Investigate native AMD FP8 handling and GPU-specific kernel tuning after
   profiling. The current emulated FP8 path also applies to the RTX 3060.

Many Compose/docs files had pre-existing user edits or were untracked before
this task, including deleted old Compose files. Preserve them. No commits,
resets, PRs, goals, or sub-agents were created.
