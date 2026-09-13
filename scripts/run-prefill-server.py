"""Start an isolated cached-image server for the local prefill reuse experiment."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--graph-verify", action="store_true")
    p.add_argument("--kv", type=int, default=65536)
    a = p.parse_args()
    assert a.name.startswith("freetoken-hip-prefill-")
    assert not a.verify or a.reuse
    root = Path(__file__).resolve().parents[1]
    out = root / "benchmarks/results/rocm-prefill-reuse"
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "-d", "--name", a.name, "--device", "/dev/kfd",
           "--device", "/dev/dri", "--shm-size", "4g", "-p", "127.0.0.1:1920:1919",
           "-v", "freetoken_rocm-kernels:/root/.cache",
           "-v", "/home/eric/.cache/huggingface:/root/.cache/huggingface:ro"]
    site = "/opt/freetoken-venv/lib/python3.12/site-packages/freetoken"
    # Mount only candidate files, preserving the image's compiled extensions.
    for path in ("kernel/batch_memcpy.py", "kernel/csrc", "moe/offload_cache.py", "server/args.py"):
        cmd += ["-v", f"{root / 'python/freetoken' / path}:{site}/{path}:ro"]
    env = {"HIP_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1",
           "MODEL_PATH": "/root/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5",
           "SERVED_MODEL_NAME": "qwen3.6:35b-nvfp4", "MEMORY_RATIO": "0.85",
           "MAX_RUNNING_REQUESTS": "2", "MAX_PREFILL_LENGTH": "2048",
           "KV_RESERVE_TOKENS": "65536", "MOE_BACKEND": "offload", "NVFP4_BACKEND": "triton",
           "FT_EXTRA_ARGS": "--expert-load serial" + (f" --num-tokens {a.kv}" if a.kv else ""),
           "TVM_FFI_CACHE_DIR": "/root/.cache/tvm-ffi-prefill-reuse-v1"}
    if a.diagnostics or a.verify or a.graph_verify:
        cmd += ["-v", f"{root / 'scripts/prefill-validation'}:/probe:ro"]
        env["PYTHONPATH"] = "/probe"
        if a.diagnostics:
            env["FREETOKEN_PREFILL_DIAGNOSTICS"] = "1"
        if a.verify:
            env["FREETOKEN_PREFILL_VERIFY"] = "1"
        if a.graph_verify:
            cmd += ["-v", f"{root / 'scripts/graph-validation'}:/graphprobe:ro",
                    "-v", f"{out}:/results"]
            env["FREETOKEN_VALIDATE_GRAPHS"] = "1"
            env["FREETOKEN_GRAPH_TRACE_PREFIX"] = a.name
    for name, value in env.items():
        cmd += ["-e", f"{name}={value}"]
    cmd += ["freetoken:rocm-hip-graphs", "serve", "--attention-backend", "triton",
            "--moe-cache-auto", "--cuda-graph-max-bs", "2"]
    if a.reuse:
        cmd += ["--moe-prefill-hit-d2d"]
    (out / f"{a.name}-launch.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
