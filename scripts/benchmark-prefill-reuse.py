"""Run isolated reuse-off/on/on/off model benchmarks using cached local images.

The caller must free AMD model memory first. This driver only starts/stops its
own uniquely named test containers, and always stops each model in a finally.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def swap_counters():
    return {k: int(v) for k, v in (s.split() for s in Path("/proc/vmstat").read_text().splitlines())
            if k in ("pswpin", "pswpout")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix")
    p.add_argument("--kv", type=int, default=65536)
    p.add_argument("--runs", type=int, default=5)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "benchmarks/results/rocm-prefill-reuse"
    for index, reuse in enumerate((False, True, True, False)):
        name = f"freetoken-hip-prefill-{a.prefix}-{index}-{'on' if reuse else 'off'}"
        start = [sys.executable, str(root / "scripts/run-prefill-server.py"), name, "--kv", str(a.kv)]
        if reuse:
            start.append("--reuse")
        print("START", name, flush=True)
        subprocess.run(start, check=True)
        try:
            for _ in range(180):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:1920/health", timeout=3) as response:
                        if json.load(response)["status"] == "ok":
                            break
                except Exception:
                    pass
                state = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Running}}", name], text=True).strip()
                if state != "true":
                    raise RuntimeError(name + " stopped during startup")
                time.sleep(2)
            else:
                raise TimeoutError(name)
            before = swap_counters()
            subprocess.run([sys.executable, str(root / "scripts/measure-prefill-reuse.py"),
                            "--bench", "--runs", str(a.runs), "--output", str(out / (name + ".json"))], check=True)
            (out / (name + "-swap.json")).write_text(json.dumps({"before": before, "after": swap_counters()}, indent=2) + "\n")
        finally:
            subprocess.run(["docker", "stop", name], check=True)
            with (out / (name + "-server.txt")).open("w") as f:
                subprocess.run(["docker", "logs", name], stdout=f, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
