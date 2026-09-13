"""Send controlled sequential diagnostic requests to an already running server."""

import argparse
import json
from pathlib import Path
import time
import urllib.request


def api(base, path, body=None):
    req = urllib.request.Request(base + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def streamed(base, body):
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = last = None
    text, usage = [], None
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    now = time.monotonic()
                    first = now if first is None else first
                    last = now
                    text.append(content)
    assert first is not None and usage is not None
    return {"ttft_s": first - start, "elapsed_s": time.monotonic() - start,
            "decode_tok_s": (usage["completion_tokens"] - 1) / (last - first),
            "usage": usage, "text": "".join(text)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:1920")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--runs", type=int, default=10)
    a = p.parse_args()
    result = {"before": api(a.base, "/v1/cache/status"), "requests": []}
    # Early unique text prevents long shared-prefix KV hits. The first request
    # observes startup residency; its 600 decode steps deliberately warm experts.
    cases = [(0, 40, 600), (1, 40, 64), (2, 700, 64), (3, 1400, 64)]
    if a.bench:
        cases = [(0, 40, 600)] + [(100 + group * 1000 + i, repeats, 64)
                 for group, repeats in enumerate((40, 155)) for i in range(a.runs)]
        cases += [(200 + group * 2000 + i, repeats, 64)
                  for group, repeats in enumerate((630, 1260)) for i in range(2)]
    for index, repeats, output in cases:
        prompt = f"Unique document {index}: " + (
            f"Section {index} describes a garden with green plants and plentiful sunlight. " * repeats
        ) + "Explain the document in detail."
        body = {"model": "qwen3.6:35b-nvfp4", "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": output, "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        start = time.monotonic()
        response = (streamed(a.base, body) if a.bench else
                    api(a.base, "/v1/chat/completions", body))
        result["requests"].append({"index": index, "elapsed_s": time.monotonic() - start,
                                   "response": response})
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(index, response.get("usage"), response.get("ttft_s"), flush=True)
    if a.bench:
        # Rebuild just the expert cache at the same capacity after kernels have
        # warmed. This separates cold residency from first-request JIT/autotuning.
        result["cold_requests"] = []
        for index in range(3):
            rebuild = api(a.base, "/v1/cache/rebuild", {
                "moe_cache_size": result["before"]["geometry"]["moe_cache_size"]})
            assert rebuild["status"] == "ok", rebuild
            prompt = f"Cold document {index}: " + (
                "Section describes a garden with green plants and plentiful sunlight. " * 40
            ) + "Explain the document in detail."
            body = {"model": "qwen3.6:35b-nvfp4", "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": 64, "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            response = streamed(a.base, body)
            result["cold_requests"].append({"rebuild": rebuild, "response": response})
            a.output.write_text(json.dumps(result, indent=2) + "\n")
            print("COLD", index, response["usage"], response["ttft_s"], flush=True)
    result["after"] = api(a.base, "/v1/cache/status")
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
