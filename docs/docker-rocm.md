# Linux Docker on Radeon AI PRO R9700

Experimental single-GPU ROCm support for RDNA4 (`gfx1201`). Start with the
small dense BF16 checkpoint `Qwen/Qwen3-0.6B` before testing larger models.
This is a ROCm/HIP port of FreeToken, not a Vulkan backend.

The standalone `compose.rocm.yaml` uses port 1920 and `ROCM_*` settings so an
existing CUDA deployment's `.env` does not select an NVIDIA model or claim
its port. Local NVIDIA deployment files are not included in this fork.

## Requirements

- Linux with the AMD kernel driver and working `/dev/kfd` and `/dev/dri`.
- Docker Engine with Compose v2.
- An R9700 (`gfx1201`, 32 GB). No architecture spoofing is required.
- Disk space for the development image and a separate PyTorch environment.

The image uses the ROCm 7.2.2 development toolchain from AMD's PyTorch image,
then installs `torch==2.11.0+rocm7.2` in an isolated virtual environment to
match this checkout's PyTorch requirement. PyTorch supplies its matching
`triton-rocm` package. Do not install the separate CUDA `triton`
distribution over it, or use `freetoken[accel]` in the ROCm environment.

`prepare-rocm-flashlib.py` builds a local `flashlib==0.3.0+rocm` wheel with
unchanged library code. Its metadata selects `triton-rocm` and omits the
unused NVIDIA CuTe dependency. This prevents flashlib's upstream dependency
resolution from overwriting ROCm Triton. The image retains this wheel and
the resolved package list under `/opt/freetoken-wheels`.

The host supplies the kernel driver; HIP libraries, compilers, PyTorch and
Triton are supplied by the container. See [AMD's Linux support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html).

## Build and Test

```bash
docker compose -f compose.rocm.yaml build
docker compose -f compose.rocm.yaml run --rm freetoken-rocm \
  python /usr/local/bin/freetoken-rocm-smoke
```

The smoke test exercises BF16 matrix multiplication, Triton activations and
RMSNorm, exact-size pinned host memory, HIP JIT embedding lookup, KV stores,
host-to-device expert gathering, and CPU radix comparison. It requires no
model weights and fails on numerical mismatches.

## Serve

Use `.env.rocm.example` as a starting configuration. Its cache path defaults to
`${HOME}/.cache/huggingface`; override it if your weights are stored elsewhere.

```bash
docker compose --env-file .env.rocm.example -f compose.rocm.yaml up -d
docker compose -f compose.rocm.yaml logs -f freetoken-rocm
```

The API is bound to `127.0.0.1:1920` by default. Test it with:

```bash
curl --fail http://127.0.0.1:1920/v1/models
curl --fail http://127.0.0.1:1920/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

To stop only the ROCm service:

```bash
docker compose -f compose.rocm.yaml down
```

Settings include `ROCM_MODEL` (Hugging Face ID or `/models/...`),
`ROCM_MODEL_NAME`, `ROCM_DEVICE`, `ROCM_NUM_PAGES`, `ROCM_MEMORY_RATIO`,
`ROCM_MAX_REQUESTS`, `ROCM_KV_RESERVE_TOKENS`, and `ROCM_MOE_BACKEND`. `ROCM_EXTRA_ARGS` adds FreeToken
CLI options. Change `ROCM_BIND_ADDRESS` to expose the API beyond localhost.
The API has no authentication by default.

### Qwen3.6 35B NVFP4

The full-model override selects the cached NVIDIA checkpoint, the Triton NVFP4
offload backend, and automatic expert-cache sizing. It removes the small-model
`--num-pages 4096` override. Allow enough free host RAM for the complete pinned
expert banks (roughly 20 GB) plus loading overhead and the rest of the service;
checkpoint disk size is not its peak RAM requirement. Do not start a second copy
alongside another large service without checking available RAM.

```bash
docker compose --env-file .env.rocm.example \
  -f compose.rocm.yaml -f compose.rocm.qwen35.yaml up -d freetoken-rocm
```

For three simultaneous requests with up to 262144 total tokens each, the
aggregate reservation is 786432 tokens, and concurrency is configured separately:

```bash
ROCM_MAX_REQUESTS=3 ROCM_KV_RESERVE_TOKENS=786432 \
  docker compose --env-file .env.rocm.example \
  -f compose.rocm.yaml -f compose.rocm.qwen35.yaml up -d freetoken-rocm
```

This allocation and short-request concurrency have been tested on this server;
three fully populated 256K contexts have not been benchmarked.
The reserve is a shared minimum KV capacity before auto-sizing the expert cache,
not a per-request partition. Input and generated output share the model's 262144
token sequence limit. Leave allocation headroom and room for output tokens.

This checkpoint has 10 full-attention layers, 2 KV heads, and head dimension 256.
The BF16 KV cost is `10 * 2 * 2 * 256 * 2 = 20480` bytes/token: 15 GiB for three
256K contexts, excluding recurrent state, weights, expert cache, and workspace.
This Qwen/Triton path has no separate Q8 KV option in this checkout. A llama.cpp
Q8 KV budget is therefore not interchangeable with this BF16 budget, even though
the weights themselves are NVFP4/FP8. More host RAM increases offload capacity;
it does not increase PCIe bandwidth. Measure per-request TTFT and decode latency,
aggregate tokens/sec, and cache misses at each concurrency/cache split.

### Validation on This Server

Validated on the R9700 (`gfx1201`) with PyTorch 2.11.0+rocm7.2 / Triton 3.6.0:

- Native smoke test, including pinned host-to-GPU expert gathering.
- Pinned-memory and Triton attention numerical tests.
- NVFP4 expert prefill/decode, overlapped transfers, and cache replacement tests.
- NVFP4 dense projections in both weight layouts at batch sizes 1, 4, and 64.
- Per-tensor FP8 W8A16 projections, including batched decode and prefill.
- NVFP4 CPU executor against GPU/dequantized references (batch sizes 1, 3, and 8).
- Cached Qwen3-0.6B serving on port 1920: correct ordinary and streaming responses.
- Cached Qwen3.6-35B-A3B-NVFP4: correct initial response and llama-benchy coherence
  checks, with three concurrent short requests on the R9700.

These checks are not a comprehensive model-quality evaluation. NVIDIA-specific
W8A8/Marlin/b12x tests are excluded or skipped on this ROCm path.

### Initial Measurements

September 11, 2026, `llama-benchy 0.4.0`, 512-token prompts, 64 requested output
tokens, two measured runs after warmup, prefix caching disabled in the client,
thinking disabled. Graph capture is **off**, and the router uses the existing
pure-PyTorch fallback because optional `triton_kernels` is absent. These are
proof-of-function measurements, not tuned performance or a Vulkan comparison.

| KV Allocation | Expert Slots | Concurrent Requests | Total Decode tok/s | Per-request Decode tok/s | Mean E2E TTFT |
|---|---:|---:|---:|---:|---:|
| 6.35 GiB / 332745 tokens | 10240 | 1 | 20.75 | 20.75 | 0.74 s |
| 15.00 GiB / 786479 tokens | 4644 | 1 | 20.98 | 20.98 | 0.74 s |
| 15.00 GiB / 786479 tokens | 4644 | 2 | 40.57 | 20.51 | 1.33 s |
| 15.00 GiB / 786479 tokens | 4644 | 3 | 60.26 | 20.39 | 1.36 s |

The smaller expert cache did not materially reduce short-request decode speed
in this sample; three-request per-user decode was about 3% below the single-user
result. That does **not** predict performance with three large, unrelated contexts.
The allocated KV capacity is not the benchmark's actual context length.
Host container memory was about 21 GiB after loading; this is not peak startup RAM.

Raw results: [full expert cache](../benchmarks/results/rocm-gfx1201/freetoken-rocm-qwen35-full-cache.json),
[768K KV reservation](../benchmarks/results/rocm-gfx1201/freetoken-rocm-qwen35-768k-cache.json).
See the [validation summary](rocm-validation-summary.md) for the measured Vulkan
comparison, long-context results, and follow-up work.

Single-run long-context checks also completed with the 768K reservation:
16384 prompt tokens at 20.95 decode tok/s (8.38 s E2E TTFT), and 131000 prompt
tokens at 20.53 decode tok/s (92.01 s E2E TTFT), with 64 requested output tokens.
No benchmark sequence exceeded 128K. These are functional checks, not a
long-context quality evaluation or a multi-user long-context stress test.
The client's derived PP throughput fields are invalid for this FreeToken run
(some exceed millions of tok/s); use E2E TTFT, not those PP/TTFR estimates.
The unmodified raw JSON is retained for transparency.

Reproduce the short concurrency run against the three-request configuration:

```bash
uvx llama-benchy==0.4.0 \
  --base-url http://127.0.0.1:1920/v1 \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --pp 512 --tg 64 --depth 0 --concurrency 1 2 3 --runs 2 \
  --no-cache --latency-mode generation \
  --extra-body 'chat_template_kwargs={"enable_thinking":false}' \
  --save-result /tmp/freetoken-rocm-bench.json --format json
```

The ROCm API can be used at `http://127.0.0.1:1920/v1` while its container is
running. This server's existing `llama-swap` also uses the R9700; do not load a
competing model during a benchmark. Stop only the ROCm service before restoring it:

```bash
docker compose -f compose.rocm.yaml stop freetoken-rocm
docker start llama-swap
```

## Reusing and Archiving Dependencies

Source edits reuse Docker's dependency layers. The named `rocm-deps` build
stage also lets you preserve the full installed toolchain, ROCm/PyTorch
environment, and Python dependencies in a tagged image and a checksummed
archive. This retains installed binaries; the current build does not keep
the original downloaded wheels.
Subsequent application dependency downloads use a persistent BuildKit pip
cache (`freetoken-rocm-pip`) as well.
The completed dependencies on this server are also tagged `freetoken:rocm-deps`;
a separate archive has not yet been exported.

After the initial build completes:

```bash
bash scripts/archive-rocm-assets.sh export ./archives/rocm
ROCM_DEPS_IMAGE=freetoken:rocm-deps docker compose -f compose.rocm.yaml build
```

The export reuses the completed dependency stage and compresses the image
with zstd. It can take several minutes and requires substantial free space.
`archives/` is excluded from Git and the Docker build context. Keep the
archive on durable storage for recovery after Docker cache/image pruning.

Restore on this or another compatible Linux Docker host:

```bash
bash scripts/archive-rocm-assets.sh restore ./archives/rocm
ROCM_DEPS_IMAGE=freetoken:rocm-deps docker compose -f compose.rocm.yaml build
```

With that override, the runtime builds directly from the restored dependency
image without reinstalling packages. Omit the override when intentionally
changing dependencies, then create a new archive in a different directory.
Model weights live separately in `ROCM_HF_CACHE` and `ROCM_MODELS_DIR`.

## Port Scope

- Native pinned-memory and CPU executor extensions link HIP rather than cudart.
- Runtime C++ kernels compile through tvm-ffi's existing Linux HIP support.
- NVIDIA-only optional packages and PDL launch options are excluded on ROCm.
- Triton activation math and grouped decode tiles have AMD-compatible paths.
- E4M3 scales use the existing portable decoding path on ROCm; AMD capability
  numbers must not be interpreted as NVIDIA SM versions.
- The expert-copy kernel uses HIP loads/stores in place of PTX instructions.
- Graph capture defaults off (`ROCM_GRAPH_MAX_BS=0`) for initial validation.
  Enable it only after testing capture and repeated replay on the target model.

Multi-GPU, all quantization formats, and full MoE model coverage are not implied
by a successful dense-model or kernel smoke test. The existing PyNCCL,
FlashInfer, and CUDA batch-copy paths require separate porting/validation.
Vulkan performance from llama.cpp is a useful comparison, but switching
FreeToken to Vulkan would also require replacing its PyTorch/Triton/native
kernel execution paths.

## Provenance

The HIP launch adapter, AMD activation math, RDNA attention tile adjustment,
and expert-copy adaptation are based on the approach in
[Maxritz/FreeToken-ROCm](https://github.com/Maxritz/FreeToken-ROCm), inspected
at commit `89f14790de7bbc69dd61a999ee76b86ba8316f26` (Apache-2.0), linked from
[upstream issue #82](https://github.com/FlashML-org/FreeToken/issues/82).
Windows-specific Python package patches and IPC changes are not needed here.
