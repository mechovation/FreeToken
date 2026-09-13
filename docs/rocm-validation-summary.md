# R9700 FreeToken Proof of Function

**September 13 update:** HIP decode graphs are validated on Qwen3.6 NVFP4.
At a fixed 65536-token KV allocation with all 10240 expert slots, graphs improve
single-request decode **20.63 → 74.67 tok/s** and two-request aggregate decode
**39.93 → 106.94 tok/s**. Ten measured samples per concurrency/configuration,
including reverse-order repeats. Near-65K prompts also pass at 52.99 decode
tok/s and 36.10 s TTFT. See the
[graph implementation, validation and results](rocm-hip-graphs-results.md).
The graph-disabled measurements below describe the initial port.
After the September 13 tests, AMD FreeToken was stopped and the previously
running NVIDIA FreeToken and AMD llama-swap/Qwen model were restored.

Tested September 11, 2026. Linux Docker on Radeon AI PRO R9700, `gfx1201`,
32 GB VRAM. No NVIDIA GPU was used for the ROCm tests.

## Outcome

- Qwen3.6-35B-A3B-NVFP4 loads and serves through ROCm with expert offload.
- Numerical tests cover attention, pinned transfers, NVFP4 expert/dense math,
  FP8 projections, and the CPU executor. Model coherence checks passed.
- Three short concurrent requests worked. A single near-128K request completed.
- A 768K aggregate BF16 KV reservation fits by reducing GPU expert-cache capacity.
- This initial port is substantially slower than the existing Vulkan deployment.
- State after September 11 tests: ROCm FreeToken stopped, llama-swap restored.
  The NVIDIA FreeToken container had been stopped by the user. See the later
  updates below for subsequent tests and service state.

## Short Benchmark Comparison

Client: llama-benchy 0.4.0. 512-token prompts, 64 requested generated tokens,
two measured runs after warmup, client prefix caching disabled, thinking off.
The client used the cached NVIDIA Qwen tokenizer for both endpoints.

| Backend | Requests | Total Decode tok/s | Per-request Decode tok/s | Mean E2E TTFT |
|---|---:|---:|---:|---:|
| FreeToken ROCm | 1 | 20.98 | 20.98 | 0.74 s |
| FreeToken ROCm | 2 | 40.57 | 20.51 | 1.33 s |
| FreeToken ROCm | 3 | 60.26 | 20.39 | 1.36 s |
| llama-swap / Vulkan | 1 | 110.08 | 110.08 | 0.29 s |
| llama-swap / Vulkan | 2 | 171.42 | 85.73 | 0.55 s |

FreeToken: ROCm 7.2.2 toolchain, PyTorch 2.11.0+rocm7.2, Triton 3.6.0,
`nvidia/Qwen3.6-35B-A3B-NVFP4`, BF16 KV, 4644 expert slots, 786479 KV tokens,
graph capture disabled, pure-PyTorch router fallback.
With all 10240 expert slots and 332745 KV tokens, single-request decode was
20.75 tok/s: no meaningful short-workload penalty from reducing the cache here.

Vulkan: unchanged `qwen3.6:35b` configuration, llama.cpp build 10423 / `a94d563ed`,
`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` from unsloth, Q8_0 K and V cache,
flash attention enabled, 99 GPU layers requested, 524288 total context, two slots.
The user reports approximately 150 tok/s in their usual Vulkan tests; the
matched short test above measured 110 tok/s. No settings were changed to chase
that reported number. Three clients were not benchmarked against its two slots.

This matches client workload, not quantization or engine configuration. It does
not isolate ROCm versus Vulkan from kernel, graph, weight, or KV-format differences.

## Longer Context and Memory

FreeToken, concurrency one, one measured run each, 64 requested output tokens:

| Prompt Tokens | Decode tok/s | E2E TTFT |
|---:|---:|---:|
| 16384 | 20.95 | 8.38 s |
| 131000 | 20.53 | 92.01 s |

Benchmark sequences stayed below 131072 total tokens. Allocating 768K KV capacity
does not establish performance with three fully populated 256K contexts.
BF16 KV takes 15 GiB for that aggregate capacity; this Qwen/Triton path has no
separate Q8 KV option. Host container memory after loading was about 21 GiB,
not a peak startup measurement. Additional RAM helps model/service capacity but
does not itself fix GPU launch overhead, PCIe bandwidth, or the measured speed gap.

Raw [results](../benchmarks/results/rocm-gfx1201/) are unmodified client output.
In the Vulkan JSON, `model` records the tokenizer/model argument; the actual API
alias was `qwen3.6:35b` via `--served-model-name`, not the NVFP4 checkpoint.
Discard the FreeToken JSON's derived PP throughput/TTFR estimates, which are
inconsistent with wall-clock prefill; the table uses its E2E TTFT metric instead.

## September 12: FP8 versus NVFP4

`Qwen/Qwen3.6-35B-A3B-FP8` also loads and passes coherence checks on the existing
ROCm image. A fresh five-repeat comparison at matched settings measured:

| Decode Workload | NVFP4 tok/s | FP8 tok/s |
|---|---:|---:|
| One request | 19.47 | 20.33 |
| Two requests, aggregate | 35.34 | 29.98 |

FP8 gave only a 4.4% single-request decode improvement, with 15.2% lower two-request
throughput. Single-request E2E TTFT increased from 0.718 s to 1.206 s, and observed
host memory increased from about 21 GiB to 34 GiB. Both used a shared 262144-token
KV floor; FP8 cached 5913 experts versus NVFP4's 10240. This is not a clear
performance upgrade for the current ROCm port.

See the [FP8 comparison](rocm-fp8-comparison.md) for raw data, exact settings,
numerical checks, model revisions, caveats, and reproduction commands. These
512-prompt/128-output tests are separate from September 11's 512/64 workload.

## September 12: llama-swap on ROCm

The newly available llama-swap deployment is confirmed to use ROCm/HIP on the
R9700: the binary lists `ROCm0`, `rocminfo` reports `gfx1201`, and the live process
loads `libggml-hip.so` and AMD runtime/BLAS libraries. The image contains ROCm
**7.2.1**, llama.cpp build **10920 / `eafe15a5e`**, not ROCm 10. The backend banner
appears with `-lv 4`; its absence at the default verbosity 3 is not a fallback.

Five warmed runs, same 512/128 client workload as the FreeToken comparison:

| Decode Workload | llama.cpp ROCm / GGUF Q4 | FreeToken ROCm / NVFP4 | FreeToken ROCm / FP8 |
|---|---:|---:|---:|
| One request, tok/s | 78.66 | 19.47 | 20.33 |
| Two requests, aggregate tok/s | 130.73 | 35.34 | 29.98 |

A separate 512/64 run measured 76.92 / 129.87 tok/s at concurrency one / two,
versus the saved Vulkan baseline's 110.08 / 171.42. ROCm decode was 30.1% / 24.2%
lower, but these deployments use different llama.cpp builds, so this is not an
isolated backend A/B. The substantial gap between the two ROCm engines also
cannot be attributed simply to ROCm versus Vulkan; quantization, KV settings,
graph replay, and kernel implementation still differ.

See the [llama-swap ROCm report](rocm-llama-swap-comparison.md) for verification
commands, exact configuration, latency results, raw data, and reproduction.
llama-swap was left running; FreeToken remained stopped and NVIDIA was untouched.

## September 12: RTX 3060 FreeToken comparison

The existing `freetoken:cuda13` image also serves the same NVFP4 checkpoint on
the RTX 3060 12GB. Five warmed 512/128 runs at concurrency one and two:

| FreeToken configuration | One request tok/s | Two requests total tok/s | One request E2E TTFT |
|---|---:|---:|---:|
| R9700, 256K KV floor, graphs off (saved baseline) | 19.47 | 35.34 | 0.718 s |
| RTX 3060, 64K KV floor, graphs on | 42.90 | 42.63 | 3.085 s |
| RTX 3060, 64K KV floor, graphs off | 22.18 | 39.30 | 3.133 s |
| RTX 3060, 256K KV floor, graphs off | 17.66 | 18.19 | 3.051 s |

The graphs-off NVIDIA cases were comparison controls, not the user's previous
NVIDIA configuration. Graphs on with a 64K shared KV floor is the normal CUDA
setup. Reducing the floor from 256K to 64K increases the 3060's GPU expert cache
from 797 to 3065 slots; the R9700 baseline cached all 10240. These capacity
settings are shared KV reservations, not the short benchmark's actual input size.

The normal NVIDIA setup decodes faster here, but first-token latency remains
substantially higher. Different graph settings, expert residency, router kernels,
and runtime builds prevent interpreting this as a GPU-only comparison. The
3060's active PCIe link reports Gen 4 x4. The CUDA client uses server-reported
completion counts (127 for a requested budget of 128); raw measurements are
preserved, with the accounting caveat explained in the full report.

The separate NVIDIA filled-context check completed a 16384-token prompt at
32.64 tok/s with 29.516 s TTFT, and a 65000-token prompt at 39.00 tok/s with
130.234 s TTFT (one measured request each after warmup). There is no saved
AMD 65000-token result; the older AMD 16384-token test measured 20.95 tok/s
and 8.378 s TTFT.

See [RTX 3060 results and reproduction](rtx3060-benchmark-comparison.md), including
raw JSON, logs, allocation details, and the separate long-context check. The
`freetoken-nvidia` service is left running at `http://127.0.0.1:1919/v1` as
`qwen3.6:35b-nvfp4`, with 64K shared KV and graphs enabled. Other services were
not reconfigured or stopped for this experiment.

## Follow-up: ROCm 10

Worth a separate test image, not an in-place replacement of the working baseline.
AMD lists R9700/gfx1201 and describes HIP graph improvements in the
[10.0.0 release notes](https://rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html).
Check the host driver against the
[compatibility matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html)
before assuming a container-only upgrade is supported.

The Dockerfile explicitly installs a ROCm 7.2 PyTorch wheel: changing `FROM` alone
will not migrate the Python runtime. Select matching PyTorch/Triton packages,
adapt the flashlib metadata pin, rebuild native extensions and JIT caches, and
rerun numerical/model tests. ROCm 10 also changes packaging/layout; consult the
[transition guide](https://rocm.docs.amd.com/en/docs-10.0.0/about/transition-guide-TheRock.html).
Profile the current stack, validate HIP graph replay, and investigate fused
routing/native RDNA FP8 paths alongside the upgrade. No speedup is assumed.
No ROCm 10 image or host-driver upgrade was attempted today.

## Reproduction and Cached Assets

See [Docker setup and benchmark commands](docker-rocm.md).
Images `freetoken:rocm-gfx1201` and `freetoken:rocm-deps` are retained locally;
source-only rebuilds reused dependencies without downloading PyTorch again.
Hugging Face model weights and compiled kernel caches are mounted separately.
`scripts/archive-rocm-assets.sh` supports a checksummed dependency-image export
and restore; the archive has not yet been exported.
