# ROCm fused router validation

September 13, 2026. The optional `triton_kernels` router from the existing CUDA
image runs unchanged on AMD with the existing ROCm/Triton stack. It passed
numerical and HIP graph replay checks, but the complete router wrapper was
slower than the PyTorch fallback in this router-only experiment. No production
router or Compose defaults were changed.

## Correctness

[`scripts/validate-fused-router.py`](../scripts/validate-fused-router.py) passed
576 cases: batches 1/2/33/512, expert counts 32/64/256/257, top-k 4 or 8,
FP32/FP16/BF16 inputs, both renormalization modes, random/tied/extreme logits,
and both unpadded and partially padded inputs. It checks FP32 weights, int32
expert IDs, contiguous output, unique selected experts, the selection cutoff,
and weights against a CPU float64 oracle (atol 2e-7, rtol 2e-5).

It also passed 100 changing-input graph replays per batch 1/2 and
renormalization mode: 400 replays total, including device-side valid-token
counts changing between zero and the full batch. Graph inputs here are FP32.

The implementations return experts in different orders. Checks compare
weights with their selected expert IDs rather than requiring positional
equality. Exact ties may select different experts. For non-renormalized
routing, the wrapper selects after FP32 softmax; extreme logits can underflow
to tied zero probabilities. An initial oracle incorrectly required selection
by original logits in that case. The corrected oracle checks the FP32
probability cutoff and still checks weights against float64 probabilities.

## Router timing

The complete FreeToken wrappers were timed with BF16 input, 256 experts and
top-k 8. Each graph contains 100 router calls; ten event-timed replays yield
per-call samples. Each comparison is repeated in reverse order. Below are
the ranges of the two medians, in microseconds, with renormalization enabled.

| Batch | PyTorch fallback | Existing fused router |
|---|---:|---:|
| 1 | 24.30–26.73 | 31.64–31.65 |
| 2 | 24.64–24.65 | 31.81–31.81 |
| 32 | 24.80–24.81 | 31.31–31.32 |
| 512 | 31.71–31.73 | 32.64–32.64 |

Raw samples and the non-renormalized measurements are in
[`validation.json`](../benchmarks/results/rocm-fused-router/validation.json).
This is a microbenchmark, not model throughput evidence. GPU clocks were not
locked, and the existing llama-swap service remained running. No 35B model
was loaded for this experiment and no service was stopped.

The source uses a streaming top-k over 32-row blocks, produces an auxiliary
bitmatrix unused by FreeToken, and returns int16 IDs that the wrapper converts
to int32. These are candidates for a focused decode optimization, not proven
causes of the measured slowdown. The next experiment should benchmark a
router that directly emits the two tensors FreeToken needs for batches 1/2,
then validate full-model outputs and throughput before changing defaults.

## Reproduction and provenance

The router is source bundled into the existing CUDA image, with no separate
installed distribution version. No packages were downloaded or upgraded.

- CUDA source image: `freetoken:cuda13`, image ID
  `5ec8d3b85067632b64c5a20ce2b02b06eff95d3800854344fa6310dd5af9b119`.
- AMD runtime: `freetoken:rocm-hip-graphs`, image ID
  `e03edc33baca7fdbb5eb1c88b79bc406e373833c494d80a7238b7a4d69101340`.
- PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`, Triton `3.6.0`.
- Router source SHA256 (sorted relative Python filenames and contents):
  `34595f3efac3cc59e3e5daf5b4a9e7430907c5e1526951a0693e321351d19764`.

From the repository root, with the cached images available:

```bash
mkdir -p /tmp/freetoken-router-source
docker run --rm --entrypoint tar freetoken:cuda13 \
  -C /opt/freetoken-venv/lib/python3.12/site-packages \
  -cf - triton_kernels > /tmp/freetoken-router-triton-kernels.tar
tar -xf /tmp/freetoken-router-triton-kernels.tar -C /tmp/freetoken-router-source
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --entrypoint python \
  -v /tmp/freetoken-router-source:/router:ro \
  -v "$PWD":/workspace/FreeToken \
  -v /tmp/freetoken-router-cache:/root/.triton \
  -e PYTHONPATH=/router:/workspace/FreeToken/python \
  freetoken:rocm-hip-graphs \
  /workspace/FreeToken/scripts/validate-fused-router.py \
  --output /workspace/FreeToken/benchmarks/results/rocm-fused-router/validation.json
```

The extracted source and compiler cache are temporary local artifacts; the
image IDs and source hash identify what was tested. The script raises on any
failed correctness check before writing a success report.
