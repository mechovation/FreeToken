"""Validate reuse serving, graph/eager parity, and cache lifecycle at auto sizing."""

import concurrent.futures
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="freetoken-hip-prefill-lifecycle")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("graph_bench", root / "scripts/benchmark-hip-graphs.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.OUT = root / "benchmarks/results/rocm-prefill-reuse"
    name = args.name
    subprocess.run([sys.executable, str(root / "scripts/run-prefill-server.py"), name,
                    "--reuse", "--kv", "0", "--graph-verify"], check=True)
    try:
        bench.ready(name)
        initial = bench.api("/v1/cache/status")
        assert initial["geometry"]["num_pages"] > 65536
        assert initial["geometry"]["moe_cache_size"] == 10240
        responses = [bench.chat(900, 96)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            responses.extend(pool.map(lambda i: bench.chat(i, 96), (901, 902)))
        assert all(r["choices"][0]["message"]["content"] for r in responses)
        bench.save(name + "-auto-requests.json", responses)
        bench.save(name + "-auto-stats.json", bench.api("/v1/stats"))
        # Leave capacity for the later test that increases recurrent-state slots.
        result = bench.api("/v1/cache/rebuild", {"num_pages": 65536})
        assert result["status"] == "ok", result
        bench.stress(name)
        bench.prefix_reuse(name)
        steady = []
        for i in range(6):
            assert bench.chat(950 + i, 32)["choices"][0]["message"]["content"]
            steady.append(bench.api("/v1/stats"))
        bench.save(name + "-steady-stats.json", steady)
        log = subprocess.check_output(["docker", "logs", name], stderr=subprocess.STDOUT, text=True)
        assert "MoE prefill hit-D2D enabled (HIP batch memcpy)" in log
        for bs in (1, 2):
            records = [json.loads(s.split("GRAPH_VALIDATION ", 1)[1])
                       for s in log.splitlines() if "GRAPH_VALIDATION " in s]
            assert any(r["batch_size"] == bs and r["steps"] == 32 for r in records)
        print("PASS auto sizing, both graph/eager oracles, serving and lifecycle", flush=True)
    finally:
        subprocess.run(["docker", "stop", name], check=True)
        bench.logs(name)


if __name__ == "__main__":
    main()
