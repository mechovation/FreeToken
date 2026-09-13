# R9700 HIP graph validation

September 12–13, 2026. Qwen3.6-35B-A3B-NVFP4 on the R9700, using
PyTorch 2.11.0+rocm7.2 and Triton 3.6.0. The existing graph runner captures
and replays on HIP. A separate graph engine or Triton router replacement
was not required.

## Implementation and correctness

The unchanged `freetoken:rocm-gfx1201` image captured batch size 1 and served a
coherent response. The candidate `freetoken:rocm-hip-graphs` retains the same
runtime and kernels, adds backend-specific capture diagnostics, and counts real
replays and eager forwards. Explicit capture failures abort startup with the
failing batch size. Prefill remains eager; the CLI flag remains
`--cuda-graph-max-bs` on both backends.

The first native-copy smoke attempt encountered a stale JIT binary in the old
cache namespace. Using a fresh namespace resolved its device-type rejection
without a kernel source change. The candidate and Compose now use
`/root/.cache/tvm-ffi-hip-graphs-v1`; included header changes are not reliably
reflected in TVM's inline-source cache key. The original image/cache remain.

Validated:

- 100 HIP replays at each batch size, changing BF16/Triton inputs, native
  mapped-host copies, expert IDs, and device-side miss counts (including zero).
- The shared GraphRunner against a CPU state-update oracle for 200 replays,
  alternating batch sizes and request slots.
- Native six-bank NVFP4 expert decode against independent dequantization and
  dense-matmul references for 100 replays at each batch size, including cold
  cache after capture, warm hits, and eviction from a cache smaller than the
  total expert population.
- Captured Triton attention against its independent reference with changing
  KV page indices, sequence lengths, query values, and positions.
- Captured GDN recurrence against the CPU mathematical reference for 100
  changing-input/slot replays at each batch size, on AMD and NVIDIA. The test
  accounts for the kernel's value-major state storage versus the reference's
  key-major layout; numerical tolerances were not relaxed.
- Full 35B model: 32 decode steps at each batch size comparing graph/eager
  logits and KV/GDN state from equivalent initial state. The same input tokens
  feed both paths; active KV writes and all GDN slots are restored between them.
  Tolerance is `rtol=atol=1e-3`; logged first/final-step logit differences were zero.
- 103 completed requests with single/concurrent decode, three successful
  KV/GDN rebuilds and recaptures, invalid resize rejection, and generation after
  a streaming request was cancelled.
- A repeated long prefix reused 4480 cached tokens and continued serving.
- RTX 3060: five shared replay/numerical checks and the existing real-server
  cache rebuild test passed with the changed graph/engine files.

The diagnostic traces contain `hipGraphLaunch` at both batch sizes:
[batch 1](../benchmarks/results/rocm-hip-graphs/hip-graph-bs1-trace.json),
[batch 2](../benchmarks/results/rocm-hip-graphs/hip-graph-bs2-trace.json).
They establish real replay; this profiler configuration does not expose every
kernel inside a HIP graph, so they are not a complete kernel-time breakdown.
Full-model diagnostics and state/rebuild evidence are in the
[validation log](../benchmarks/results/rocm-hip-graphs/freetoken-hip-validation-server.txt).
Early failed starts in that log came from the diagnostic harness (frozen config
assignment and comparing float32/BF16 without conversion), not capture defects.

Capture of sizes `[1, 2]` used approximately 0.33 GiB additional GPU memory at
262144 actual KV tokens, retaining all 10240 expert slots. After JIT warmup,
capture took about two seconds. These are deployment observations, not memory
guarantees for other models or concurrency.

## Controlled throughput measurements

The table pools ten measured runs per concurrency (five per boot in each
order). Throughput is mean ± population standard deviation; TTFT is mean seconds.

| Actual shared KV | Graphs | Single-request tok/s | Two-request aggregate tok/s | TTFT single / two |
|---|---|---:|---:|---:|
| 262144 | off | 20.79 ± 0.17 | 40.07 ± 0.45 | 0.807 / 1.429 s |
| 262144 | on, sizes 1/2 | 73.95 ± 0.45 | 106.02 ± 0.62 | 0.770 / 0.713 s |
| 65536 | off | 20.63 ± 0.24 | 39.93 ± 0.39 | 0.825 / 1.334 s |
| 65536 | on, sizes 1/2 | 74.67 ± 0.57 | 106.94 ± 0.40 | 0.668 / 0.800 s |

At fixed 256K allocation, HIP graphs improve decode **3.56×** for one request
and **2.65×** for two concurrent requests. At 64K those gains are **3.62×** and
**2.68×**. Changing the KV allocation has little effect on these short prompts
when all experts still fit. Every run retains all 10240 expert
slots. The graph-enabled two-request TTFT also benefits from batched prefill;
inspect server logs when comparing that metric.

The same candidate image ran graphs off/on, then on/off in reverse order, at
each of 262144 and 65536 **actual shared KV tokens**. `--num-tokens` pins the
allocation and `/v1/cache/status` confirms it. Each configuration retains all
10240 expert slots, memory ratio 0.85, two active requests, prefill chunk 2048,
serial loading, BF16 KV, Triton attention/NVFP4, and the pure-PyTorch router.

Client: cached llama-benchy 0.4.0 and the same tokenizer/corpus as the September
12 NVIDIA runs; 512 prompt tokens, requested output budget 128, five measured
runs per concurrency after warmup, concurrency 1 and 2, no cache/adaptation,
thinking disabled. Each pair is repeated in reverse order. Corpus slices are
randomized by the client, rather than identical paired prompt strings.
Report decode throughput and E2E TTFT; derived PP/TTFR fields are not reliable
for this server. Preserve the server's 127/128 and 63/64 output accounting.

The model checkpoint is revision `1355db6a052410cfd62085d94b58866fd0f2c3c5`;
the comparison tokenizer is revision `491c2f1ea524c639598bf8fa787a93fed5a6fbce`.
The tokenizer and weight blobs match those used in the NVIDIA report.

### Filled input context

Actual KV allocation is 65536 tokens. One measured request per prompt length
after warmup, output budget 64 (server reports 63), concurrency one:

| Prompt target | Graphs off tok/s | Graphs on tok/s | TTFT off / on |
|---|---:|---:|---:|
| 16384 | 20.65 | 67.47 | 8.245 / 8.165 s |
| 65000 | 20.70 | 52.99 | 36.217 / 36.096 s |

Decode improves at both filled-context lengths. Prefill latency changes little,
as expected from leaving prefill eager. The lower graph-enabled rate at 65K is
consistent with increased attention work, but this test does not isolate that
cost from the different generated token/expert sequences. These two
single-sample comparisons establish successful long-context execution, not a
comprehensive quality or variability assessment.

### Comparison with the saved RTX 3060 results

The NVIDIA numbers below are the September 12 measurements, not a new NVIDIA
throughput run. Both deployments enable graphs for batches 1/2, use BF16 KV,
and select Triton attention/NVFP4. AMD has 10240 expert slots and 65536 actual
KV tokens; NVIDIA has 3065 slots and 65569 actual KV tokens. Optional kernels,
router implementations, and available expert capacity still differ.

| Workload | R9700 HIP graphs | RTX 3060 CUDA graphs |
|---|---:|---:|
| 512-token prompt, one request | 74.67 tok/s | 42.90 tok/s |
| 512-token prompt, two requests aggregate | 106.94 tok/s | 42.63 tok/s |
| 16384-token prompt, one request | 67.47 tok/s | 32.64 tok/s |
| 65000-token prompt, one request | 52.99 tok/s | 39.00 tok/s |
| 65000-token prompt, first token | 36.10 s | 130.23 s |

The short-workload rates are **1.74× single-request** and **2.51× aggregate**
relative to the saved 3060 deployment. This is a comparison of working
deployments, not an isolated GPU architecture experiment.

All eight AMD configuration boots completed without any host swap-out between
their before/after snapshots. Small system-wide swap-in counts remained (at
most 99 pages in a run). The prior NVIDIA FreeToken and AMD llama-swap services
were paused for the measurements. Raw JSON, commands, logs, geometry, source
manifest, and memory snapshots are under
[rocm-hip-graphs](../benchmarks/results/rocm-hip-graphs/).

### Actual expert residency

An untimed diagnostic enabled the existing device-side counters before graph
capture, then reset only statistics between decode windows. Every reported
window counted real replay; no duplicate eager forward was used. Both runs
retain all 10240 expert slots. The two-request case follows the single-request
case and therefore starts with an already warmed expert cache.

| Decode window | Single-request expert miss rate | Two-request expert miss rate |
|---|---:|---:|
| 1–32 | 27.29% | 3.15% |
| 33–64 | 10.69% | 0.19% |
| 65–128 | 4.54% | 0.10% |
| 129–256 | 4.26% | 0.29% |
| 257–512 | 0.91% | 0.19% |

Capacity for all experts is useful, but it does not mean every expert is
preloaded. In the last single-request window, about 0.073 of eight active
experts per layer required a copy. Allocated GPU memory stayed within roughly
5 KiB across these checkpoints, and reserved memory was constant; this is a
bounded stability observation, not a proof against every possible leak.
See the [raw windows](../benchmarks/results/rocm-hip-graphs/expert-miss-windows.json).
These measure decode misses; the prefill full-layer-copy optimization remains
disabled. The existing stats API's CPU/fetched columns apply to hybrid mode
and must not be interpreted as CPU expert execution in this GPU offload run.

## Reproduction and deployment

Stop competing GPU inference and serialize large model loads on this host.
Build the candidate using `docker/Dockerfile.hip-graphs`, then:

```bash
python3 scripts/benchmark-hip-graphs.py freetoken-hip-example-off \
  --start --kv 65536 --graphs 0 --bench --stop
python3 scripts/benchmark-hip-graphs.py freetoken-hip-example-on \
  --start --kv 65536 --graphs 2 --bench --stop
```

This local driver assumes the host's cached image, tokenizer, corpus, and
offline `uvx` environment. Its generated launch/benchmark command JSON and
raw outputs are saved under `benchmarks/results/rocm-hip-graphs/`.
Run untimed diagnostics separately:

```bash
python3 scripts/benchmark-hip-graphs.py freetoken-hip-example-validation \
  --start --kv 262144 --graphs 2 --validate --stress --stop
python3 scripts/benchmark-hip-graphs.py freetoken-hip-example-stats \
  --start --kv 65536 --graphs 2 --stats --prefix-reuse --stop
```

Full-model comparison uses `scripts/graph-validation/sitecustomize.py` through
`PYTHONPATH` with `FREETOKEN_VALIDATE_GRAPHS=1` only in a test container. The
driver mounts writable `/results` for traces. Never enable that probe for
benchmarks; the driver rejects combining diagnostics and timed measurements.

Compose selects the candidate image, graphs `[1, 2]`, two requests, and a 64K
KV **floor**. Automatic allocation can exceed that floor after all experts fit.
The final normal Compose launch passed readiness and single/concurrent serving
with **340189 actual shared KV tokens** and all 10240 experts. This differs
from the explicitly capped 64K/256K benchmark allocations above. Its log is
[saved separately](../benchmarks/results/rocm-hip-graphs/final-amd-deployment-server.txt).
`ROCM_GRAPH_MAX_BS=0` disables graphs; set
`ROCM_RUNTIME_IMAGE=freetoken:rocm-gfx1201` as well to use the prior image.
See [deployment instructions](docker-rocm.md) for startup and rollback.

Final service state: AMD FreeToken stopped after its deployment check;
NVIDIA FreeToken restored and verified ready, then stopped at the user's request;
AMD llama-swap restored, including its previously loaded Qwen Vulkan model.
Other services were unchanged. The
[service record](../benchmarks/results/rocm-hip-graphs/final-service-state.json)
preserves the validation checkpoint before the requested NVIDIA shutdown;
the [session handoff](rocm-hip-graphs-handoff.md) records the current state.

## Remaining CUDA/ROCm differences

This audit concerns the current FreeToken code and installed images, not what
the ROCm platform could support after further work.

| Area | Current AMD path | NVIDIA comparison relevance |
|---|---|---|
| Fused MoE router | `moe/fused.py` uses separate PyTorch operations because optional `triton_kernels` is absent. Its AMD compatibility has not been established either way. | The 3060 image has the fused router. A focused AMD validation/fusion experiment is a candidate follow-up. |
| Reusing resident experts during prefill | The optional `moe_prefill_hit_d2d` path relies on CUDA-specific `kernel/batch_memcpy.py`. Normal prefill still copies full layers into overlap buffers. | This optimization was disabled in both benchmark configurations. AMD's large expert cache makes it worth investigating separately. |
| Native FP8 handling | `kernel/triton/e4m3_compat.py` forces the emulated path whenever `torch.version.hip` is set. Hardware-specific native AMD handling is not validated here. | The RTX 3060 also uses the emulated path, so this did not give it an advantage in these tests. |
| Optional native kernel libraries | ROCm dispatch disables FlashInfer/SGL kernels and uses working Triton replacements for core operations. Marlin/Blackwell NVFP4 backends are separate CUDA paths. | Both measured deployments selected Triton attention/NVFP4. CUDA can still use native library implementations for smaller operations such as normalization; performance parity is unmeasured. |

Graph replay, GDN state updates, attention, and native mapped-host expert copies
now have working AMD validation. Do not describe the remaining optimizations
as proof that ROCm cannot implement them. Further optimization should begin in
a separate measured change after this graph baseline is recorded.
