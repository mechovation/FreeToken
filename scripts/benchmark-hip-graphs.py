"""Local cached-image benchmark driver. Uses the existing host model/tokenizer cache."""

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/rocm-hip-graphs"
BASE = "http://127.0.0.1:1920"


def api(path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n")


def call(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def logs(name):
    with (OUT / (name + "-server.txt")).open("w") as f:
        call(["docker", "logs", name], stdout=f, stderr=subprocess.STDOUT)


def ready(name):
    for _ in range(300):
        try:
            if api("/health").get("status") == "ok":
                save(name + "-cache.json", api("/v1/cache/status"))
                print("READY", name, flush=True)
                return
        except Exception:
            pass
        state = subprocess.check_output(
            ["docker", "inspect", "--format", "{{.State.Running}}", name], text=True
        ).strip()
        if state != "true":
            logs(name)
            raise RuntimeError(name + " exited")
        time.sleep(2)
    logs(name)
    raise TimeoutError(name)


def chat(i, tokens=80, prefix=""):
    return api(
        "/v1/chat/completions",
        {
            "model": "qwen3.6:35b-nvfp4",
            "messages": [
                {
                    "role": "user",
                    "content": prefix
                    + f"Case {i}. Explain why plants need sunlight, with a detailed concrete example.",
                }
            ],
            "temperature": 0,
            "max_tokens": tokens,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def stress(name):
    results = []
    results.append(chat(0, 96))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results.extend(pool.map(lambda i: chat(i, 96), range(1, 3)))
        results.extend(pool.map(lambda i: chat(i, 16), range(3, 103)))
    assert all(x["choices"][0]["message"]["content"] for x in results)
    save(name + "-requests.json", results)
    print("PASS 103 requests including single/concurrent decode", flush=True)
    before = api("/v1/cache/status")["geometry"]
    save(name + "-before-rebuild.json", before)
    for body in (
        {"num_pages": before["num_pages"] - 1024},
        {"num_pages": before["num_pages"]},
        {"num_mamba_slots": before["num_mamba_slots"] + 2},
    ):
        result = api("/v1/cache/rebuild", body)
        print("REBUILD", result, flush=True)
        assert result["status"] == "ok"
        assert chat(200, 16)["choices"][0]["message"]["content"]
    try:
        api("/v1/cache/rebuild", {"num_pages": 99999999})
        raise AssertionError("invalid rebuild accepted")
    except urllib.error.HTTPError as e:
        assert e.code == 503
        print("PASS invalid rebuild rejected", flush=True)
    assert chat(201, 16)["choices"][0]["message"]["content"]
    # Abort a streaming response after its first generated content; then reuse slots.
    body = {
        "model": "qwen3.6:35b-nvfp4",
        "messages": [
            {"role": "user", "content": "Write a very long story about a forest."}
        ],
        "max_tokens": 1024,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        for line in response:
            if b'"content"' in line and b"data:" in line:
                break
    assert chat(202, 16)["choices"][0]["message"]["content"]
    print("PASS cancellation and subsequent generation", flush=True)
    save(name + "-after-rebuild.json", api("/v1/cache/status"))


def bench(name, long=False):
    tokenizer = "/home/eric/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/491c2f1ea524c639598bf8fa787a93fed5a6fbce"
    cmd = [
        "uvx",
        "--offline",
        "llama-benchy==0.4.0",
        "--base-url",
        BASE + "/v1",
        "--model",
        "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "--served-model-name",
        "qwen3.6:35b-nvfp4",
        "--tokenizer",
        tokenizer,
        "--pp",
        *(["16384", "65000"] if long else ["512"]),
        "--tg",
        "64" if long else "128",
        "--exact-tg",
        "--depth",
        "0",
        "--concurrency",
        *(["1"] if long else ["1", "2"]),
        "--runs",
        "1" if long else "5",
        "--no-cache",
        "--no-adapt-prompt",
        "--latency-mode",
        "none" if long else "generation",
        "--exit-on-first-fail",
        "--extra-body",
        'chat_template_kwargs={"enable_thinking":false}',
        "--save-result",
        str(OUT / (name + ".json")),
        "--format",
        "json",
    ]
    save(name + "-command.json", cmd)
    with (OUT / (name + "-client.txt")).open("w") as f:
        call(
            cmd,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print("BENCH DONE", name, flush=True)


def prefix_reuse(name):
    prefix = (
        "Background: plants convert sunlight, water and carbon dioxide into sugars and oxygen. "
        * 300
    )
    responses = [chat(i, 32, prefix) for i in (400, 401)]
    assert all(r["choices"][0]["message"]["content"] for r in responses)
    save(name + "-prefix-requests.json", responses)
    # The server log records actual cached-token admission. Two successful
    # requests alone would not establish that prefix reuse was exercised.
    log = subprocess.check_output(
        ["docker", "logs", name], stderr=subprocess.STDOUT, text=True
    )
    import re

    reused = [int(n) for n in re.findall(r"#cached-token: (\d+)", log)]
    assert any(n > 2048 for n in reused), "prefix was not reused"
    print("PASS prefix reuse, max cached tokens:", max(reused), flush=True)


def start(name, kv, graphs, *, stats=False, validate=False):
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--shm-size",
        "4g",
        "-p",
        "127.0.0.1:1920:1919",
        "-v",
        "freetoken_rocm-kernels:/root/.cache",
        "-v",
        "/home/eric/.cache/huggingface:/root/.cache/huggingface:ro",
    ]
    env = {
        "HIP_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "MODEL_PATH": "/root/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5",
        "SERVED_MODEL_NAME": "qwen3.6:35b-nvfp4",
        "MEMORY_RATIO": "0.85",
        "MAX_RUNNING_REQUESTS": "2",
        "MAX_PREFILL_LENGTH": "2048",
        "KV_RESERVE_TOKENS": str(kv),
        "MOE_BACKEND": "offload",
        "NVFP4_BACKEND": "triton",
        "FT_EXTRA_ARGS": f"--expert-load serial --num-tokens {kv}",
    }
    if stats or validate:
        cmd += [
            "-v",
            f"{ROOT / 'scripts/graph-validation'}:/probe:ro",
            "-v",
            f"{OUT}:/results",
        ]
        env["PYTHONPATH"] = "/probe"
        env["FREETOKEN_GRAPH_TRACE_PREFIX"] = name
        env["FREETOKEN_GRAPH_STATS_ONLY" if stats else "FREETOKEN_VALIDATE_GRAPHS"] = (
            "1"
        )
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [
        "freetoken:rocm-hip-graphs",
        "serve",
        "--attention-backend",
        "triton",
        "--moe-cache-auto",
        "--cuda-graph-max-bs",
        str(graphs),
    ]
    save(name + "-launch.json", cmd)
    call(cmd)
    ready(name)


def main():
    p = argparse.ArgumentParser(
        description="Reproduce the local R9700 HIP graph validation and benchmark runs."
    )
    p.add_argument("name")
    p.add_argument("--start", action="store_true")
    p.add_argument("--kv", type=int, default=262144)
    p.add_argument("--graphs", type=int, default=2)
    p.add_argument("--stress", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--long", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument(
        "--stats",
        action="store_true",
        help="collect expert miss windows, outside throughput runs",
    )
    p.add_argument(
        "--validate", action="store_true", help="compare full-model graph/eager decode"
    )
    p.add_argument("--prefix-reuse", action="store_true")
    a = p.parse_args()
    if not a.name.startswith("freetoken-hip-"):
        p.error("test container name must begin with freetoken-hip-")
    if (a.stats or a.validate) and (a.bench or a.long or not a.start or a.graphs < 1):
        p.error(
            "diagnostics require --start and graphs, and cannot be timed benchmarks"
        )
    if a.stats and a.validate:
        p.error("run statistics and numerical comparison separately")
    try:
        if a.start:
            start(a.name, a.kv, a.graphs, stats=a.stats, validate=a.validate)
        else:
            ready(a.name)
        if a.stress:
            stress(a.name)
        if a.stats:
            responses = [chat(300, 600)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                responses.extend(pool.map(lambda i: chat(i, 600), (301, 302)))
            save(a.name + "-requests.json", responses)
        if a.prefix_reuse:
            prefix_reuse(a.name)
        if a.bench or a.long:
            bench(a.name + ("-long" if a.long else ""), a.long)
    finally:
        if a.stop:
            call(["docker", "stop", a.name])
        logs(a.name)


if __name__ == "__main__":
    main()
