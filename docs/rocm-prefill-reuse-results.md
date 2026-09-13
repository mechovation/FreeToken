# ROCm prefill expert reuse

September 13, 2026. Exploration of the existing opt-in
`--moe-prefill-hit-d2d` path on Qwen3.6-35B-A3B NVFP4. Production defaults and
the PyTorch router remain unchanged. Warm-prefill latency improved in repeated
serving measurements; the implementation remains opt-in.

## Implementation

The installed HIP 7.2 runtime provides `hipMemcpyBatchAsync`. The new binding
uses its nine-argument signature with unsupported attributes omitted. CUDA
retains its existing CUDA 13 implementation. Both paths check runtime support
and perform a real copy probe before activation; unsupported setups retain the
existing full-layer-copy fallback. Startup now reports successful activation
as well as fallback. The probe's stream waits for its destination initialization.

The cache algorithm is unchanged: a per-chunk slot snapshot identifies stable
residents above the first `2E` volatile slots; device gathers serve large-bank
hits and batch H2D copies serve coalesced misses. Small banks still cross PCIe
in full. The flag remains off by default, and no runtime image was rebuilt or
production service reconfigured. Test servers use source mounts over the cached
image, preserving its compiled extensions.

## Baseline opportunity

At 65536 actual KV tokens, expert capacity is 10240, with overlap enabled.
Each expert uses 1775616 bytes across six NVFP4 banks:
1048576, 131072, 2048, 524288, 65536 and 4096 bytes per row.
Each full layer copies 454557696 bytes; 40 layers copy 18.18 GB per chunk.

The startup/cold chunks had no reusable rows. After a 599-token decode, 7199
experts were reusable outside the volatile double buffers, allowing 11.32 GB
of large-bank H2D traffic to be avoided per chunk. Later windows reached
7250 reusable experts and 11.40 GB avoidable traffic. Consecutive prefill
chunks preserved this stable residency.

Diagnostic copy-stream event intervals totaled about 649 ms per warm chunk.
Explicit compute-stream event waits were often much smaller because copies,
host submission and compute overlap. These intervals are not additive
end-to-end latency savings. The probe synchronizes intentionally and was
disabled during throughput runs.

See [baseline summary](../benchmarks/results/rocm-prefill-reuse/baseline-summary.json)
and [raw diagnostics](../benchmarks/results/rocm-prefill-reuse/baseline-server.txt).

## Copy experiment

The isolated one-layer benchmark uses production NVFP4 bank byte sizes and
256 experts. Wall time includes the slot snapshot/synchronization, host run
construction, gathers, miss submission and completion. Each configuration has
ten samples, repeated in reverse backend order. It tests contiguous and random
fragmented misses and checks the resulting bank bytes.

| Resident fraction | Full-layer copy | HIP batch reuse |
|---|---:|---:|
| 0% | about 16.5 ms | about 16.7 ms |
| 25% | about 16.5 ms | 13.4–13.9 ms |
| 75% | about 16.5 ms | 6.9–7.6 ms |
| 100% | about 16.5 ms | 3.6–3.7 ms |

A PyTorch tensor-slice submission reference performed similarly. The native
HIP binding was retained because it fits the existing batch interface with a
small backend-specific change; these measurements do not establish a material
native-versus-Python speed advantage. The tensor reference also translates
existing raw-pointer descriptors back into slices, so it is not a tuned Python
implementation.

Raw samples: [copy benchmark](../benchmarks/results/rocm-prefill-reuse/copy-benchmark.json).

## Serving performance

Four clean boots in off/on/on/off order, with diagnostics disabled, 65536 actual
KV tokens and 10240 expert slots throughout. Every boot first completed a
599-token decode to warm expert residency. Fresh prompt prefixes prevented
KV-prefix hits in the measured requests; server logs confirm zero cached tokens.
Each measured response generated 63 tokens. The table gives median streaming
TTFT, measured at the first nonempty content event.
First samples at new prompt shapes were often slower and were retained; no
outliers were discarded.

| Actual prompt tokens | Samples per mode | Reuse off | Reuse on | TTFT reduction |
|---|---:|---:|---:|---:|
| 625 | 10 | 0.669 s | 0.384 s | 42.5% |
| 2506 | 10 | 1.550 s | 1.129 s | 27.1% |
| 9475 | 4 | 4.345 s | 3.676 s | 15.4% |
| 20186 | 4 | 9.282 s | 8.165 s | 12.0% |

Decode rates were essentially unchanged: off/on medians were 73.96/74.14,
73.05/73.04, 69.58/69.56 and 65.45/65.59 tokens/s respectively. All 15 response
texts per boot, including warmup, matched the first baseline run exactly.
No swap-out occurred during the measured intervals; the last two boots had
2 and 6 pages of system-wide swap-in. GPU clocks were not locked.

The first request in each process took 20.28–20.64 seconds to first content,
including shape-specific compilation/autotuning; it is excluded from the warm
comparison. Cold-residency checks were added to the final on/off pair: after
kernel warmup, the expert cache was rebuilt at the same capacity before each
463-token request. Three off samples were 0.729/0.673/0.672 s; on samples were
0.732/0.676/0.861 s. The first two pairs were close, but one reuse sample was
slower. This small sample does not establish cold-cache performance parity.
No adaptive hit-rate cutoff was introduced.

See [serving summary and variability](../benchmarks/results/rocm-prefill-reuse/serving-summary.json)
and the per-boot requests, launch manifests, swap counters and logs alongside it.
The benchmark client now includes cold checks in every boot on reproduction;
the recorded initial run added them only for its final pair.

## Correctness evidence

- Eight reuse tests passed on AMD and RTX 3060. New NVFP4-sized fixtures exercise
  actual large-bank gathers; the older small BF16 fixtures only exercised the
  whole-small-bank path. Coverage includes mixed/all/zero hits, changing expert
  sets, poisoned volatile residents, buffer wraparound, repeated chunks, and
  rebuilds between spare capacity and the exact `2E` threshold.
- The final AMD run passed 11 reuse/shared graph tests, including changing-input
  graph replay. See [test output](../benchmarks/results/rocm-prefill-reuse/amd-tests.txt).
- Full-model validation passed 720 per-layer checks across warmup and requests
  of 543, 9123 and 18223 prompt tokens. Every materialized bank matched its host
  source exactly. Recomputing experts from a fresh full-layer host copy, using
  the same hidden states and routing, matched the reuse output at the existing
  atol/rtol of 1e-3. No tolerances were relaxed.
- All four diagnostic requests produced exactly the same message objects with
  baseline and reuse, including the initial 599-token decode. Graph-based
  decode continued after each prefill.

See [verification log](../benchmarks/results/rocm-prefill-reuse/verified-server.txt)
and [responses](../benchmarks/results/rocm-prefill-reuse/verified-requests.json).
This oracle isolates the changed expert movement/compute path; it is not a
second full-model KV/GDN-state replay oracle.

The separate lifecycle run then passed at normal auto sizing: 340189 actual KV
tokens and 10240 expert slots. It completed 32 graph/eager comparisons at each
batch size 1/2, checking logits, KV writes and every convolution/recurrent state
slot. Logged checkpoints had zero maximum logit difference.

Serving checks included 103 short single/concurrent requests, successful
rebuilds/recaptures, invalid-resize rejection, cancellation recovery, and 4480
tokens of actual prefix reuse. Auto-sized capacity was reduced to 65536 before
the lifecycle cases that increase recurrent-state capacity. Separate cold-cache
timings also exercised six expert-cache rebuilds at unchanged capacity.
After lifecycle warmup, six successive short requests reported the same
23806869504 bytes of reserved GPU memory. All experimental containers are stopped.

See [lifecycle results](../benchmarks/results/rocm-prefill-reuse/lifecycle-retry-client.txt)
and [validation summary](../benchmarks/results/rocm-prefill-reuse/validation-summary.json).

The first attempt to run the separate full-model graph/eager oracle at normal
auto-sized KV capacity exhausted VRAM inside `torch.testing.assert_close` on
the full recurrent-state tensor. The saved
[failure](../benchmarks/results/rocm-prefill-reuse/lifecycle-initial-failure.txt)
shows an allocation failure, not a numerical mismatch. The oracle now compares
every state layer individually with identical tolerances, limiting comparison
scratch memory. This is a diagnostic-only change.

## Recommendation and limits

Retain the PyTorch router and keep prefill reuse opt-in. The HIP implementation
is useful for warmed expert caches on this tested model/GPU: its repeated
TTFT gains are substantially larger than the observed warm-run variability.
Cold-cache variability needs more sampling before making reuse a default.
Smaller expert caches, other model/quantization formats and contexts beyond
the measured 20186 tokens do not have equivalent serving-performance evidence.
Concurrent serving is a correctness/lifecycle check, not a timed concurrency
comparison in this experiment.

To try it in a runtime built from these sources, add `--moe-prefill-hit-d2d`.
For Compose, append that flag to `ROCM_EXTRA_ARGS` while preserving any existing
arguments. Look for `MoE prefill hit-D2D enabled (HIP batch memcpy)` in the log.
Removing the flag and restarting restores full-layer copying. The cached
production image has not been updated by these experiments, so adding the flag
to that unchanged image alone does not install the new HIP binding.

## Reproduction and controls

The plan is [rocm-prefill-reuse-plan.md](rocm-prefill-reuse-plan.md).
Exact runtime image IDs, model snapshot, source hashes and launch commands are
saved under [the artifact directory](../benchmarks/results/rocm-prefill-reuse/).
The environment is the existing PyTorch 2.11.0+rocm7.2 / HIP 7.2.26015 stack;
no model, PyTorch or ROCm upgrade was performed.

With AMD model memory freed, from the repository root:

```bash
# Use unique test names; these scripts do not replace existing containers.
python3 scripts/run-prefill-server.py freetoken-hip-prefill-check --reuse --verify
# Wait for /health to report status=ok, then:
python3 scripts/measure-prefill-reuse.py --output /tmp/prefill-check.json
docker stop freetoken-hip-prefill-check

# Four isolated boots: off/on/on/off; five samples per short case per boot.
python3 scripts/benchmark-prefill-reuse.py repeat1 --runs 5

# Separate diagnostic run at auto sizing; includes graph/eager and lifecycle checks.
python3 scripts/validate-prefill-lifecycle.py --name freetoken-hip-prefill-lifecycle-repeat1
```

The benchmark driver stops its own model after each run. It does not manage
llama-swap. At the user's direction, AMD llama-swap remains stopped rather than
being restored. Serialize 35B loads across GPUs because they share host RAM.

An early ctypes API probe accidentally loaded the system HIP library alongside
PyTorch's wheel runtime, producing invalid-stream-handle errors. Selecting
PyTorch's already loaded runtime fixed the harness and all tested copy sizes
passed; see [API probe](../benchmarks/results/rocm-prefill-reuse/hip-api-probe.json).
An initial source-overlay test also hid the image's compiled extensions; merging
candidate sources into the installed package fixed that harness setup.
