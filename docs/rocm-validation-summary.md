# R9700 FreeToken Proof of Function

Tested September 11, 2026. Linux Docker on Radeon AI PRO R9700, `gfx1201`,
32 GB VRAM. No NVIDIA GPU was used for the ROCm tests.

## Outcome

- Qwen3.6-35B-A3B-NVFP4 loads and serves through ROCm with expert offload.
- Numerical tests cover attention, pinned transfers, NVFP4 expert/dense math,
  FP8 projections, and the CPU executor. Model coherence checks passed.
- Three short concurrent requests worked. A single near-128K request completed.
- A 768K aggregate BF16 KV reservation fits by reducing GPU expert-cache capacity.
- This initial port is substantially slower than the existing Vulkan deployment.
- Final service state: ROCm FreeToken stopped, llama-swap restored. The NVIDIA
  FreeToken container was stopped by the user and remains stopped.

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
