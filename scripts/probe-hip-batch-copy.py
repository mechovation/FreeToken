"""Probe the installed HIP batch API without changing FreeToken dispatch."""

import ctypes as C
import json
from pathlib import Path

import torch


def main():
    torch.cuda.init()
    # Use PyTorch's loaded runtime: loading /opt/rocm's second runtime gives
    # invalid-resource-handle errors for streams created by the wheel runtime.
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "libamdhip64.so" in line}
    assert len(paths) == 1, paths
    runtime = paths.pop()
    hip = C.CDLL(runtime)
    fn = hip.hipMemcpyBatchAsync
    fn.restype = C.c_int
    fn.argtypes = [C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
                   C.POINTER(C.c_size_t), C.c_size_t, C.c_void_p,
                   C.POINTER(C.c_size_t), C.c_size_t,
                   C.POINTER(C.c_size_t), C.c_void_p]
    hip.hipGetErrorString.restype = C.c_char_p
    records = []
    for size in (16, 1024, 1024 * 1024):
        src = torch.randint(0, 256, (4, size), dtype=torch.uint8).pin_memory()
        dst = torch.zeros_like(src, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        srcs = (C.c_void_p * 4)(*(src[i].data_ptr() for i in (3, 1, 2, 0)))
        dsts = (C.c_void_p * 4)(*(dst[i].data_ptr() for i in range(4)))
        sizes = (C.c_size_t * 4)(*([size] * 4))
        fail = C.c_size_t(2**64 - 1)
        status = fn(dsts, srcs, sizes, 4, None, None, 0,
                    C.byref(fail), stream.cuda_stream)
        stream.synchronize()
        record = {"bytes_per_copy": size, "status": status,
                  "error": hip.hipGetErrorString(status).decode(),
                  "failure_index": fail.value}
        if status == 0:
            record["correct"] = torch.equal(dst.cpu(), src[[3, 1, 2, 0]])
            assert record["correct"]
        records.append(record)
    print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip, "runtime": runtime,
                      "copies": records}, indent=2))


if __name__ == "__main__":
    main()
