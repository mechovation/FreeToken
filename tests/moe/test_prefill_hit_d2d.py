"""Prefill hit-D2D split: resident experts must be gathered device-side, misses
H2D'd via cudaMemcpyBatchAsync, and the buffer must end up byte-identical to the
full-layer copy for every mix -- including experts resident in the volatile
buffer slots (< 2 * num_experts), which must be re-fetched over PCIe."""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
JIT = pytest.mark.skipif(
    os.getenv("FREETOKEN_DISABLE_JIT", "").strip().lower() in {"1", "true", "yes", "on"},
    reason="batch_memcpy has no AOT prebuild; needs runtime JIT",
)


def _cuda_at_least(major: int, minor: int) -> bool:
    cuda = torch.version.cuda
    if cuda is None:
        return False
    return tuple(int(x) for x in cuda.split(".")[:2]) >= (major, minor)

BATCH_API = pytest.mark.skipif(
    not (_cuda_at_least(13, 0) or (
        torch.version.hip is not None
        and tuple(int(x) for x in torch.version.hip.split(".")[:2]) >= (7, 2)
    )), reason="batch memcpy needs CUDA >= 13.0 or HIP >= 7.2"
)

NUM_LAYERS, E, CACHE_SIZE = 3, 8, 24  # hit region = slots [16, 24)


def _make_cache() -> tuple[OffloadMoeCache, dict[str, list[torch.Tensor]]]:
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=CACHE_SIZE,
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    return cache, sources


def _seed_resident(cache, sources, layer_id: int, expert_id: int, slot: int) -> None:
    cache.slot_for_id[layer_id, expert_id] = slot
    cache.id_of_slot[slot] = layer_id * E + expert_id
    for name, per_layer in sources.items():
        cache.bank_caches[name][slot].copy_(per_layer[layer_id][expert_id])


@CUDA
@JIT
@BATCH_API
def test_batch_memcpy_roundtrip():
    from freetoken.kernel.batch_memcpy import batch_memcpy_jit

    rows, feat = 16, 1024
    src = torch.randint(0, 256, (rows, feat), dtype=torch.uint8).pin_memory()
    dst = torch.zeros(rows, feat, dtype=torch.uint8, device="cuda")
    perm = torch.randperm(rows)
    dst_ptrs = torch.tensor([dst[i].data_ptr() for i in range(rows)], dtype=torch.int64)
    src_ptrs = torch.tensor([src[p].data_ptr() for p in perm.tolist()], dtype=torch.int64)
    sizes = torch.full((rows,), feat, dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        batch_memcpy_jit(dst_ptrs, src_ptrs, sizes, stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(dst.cpu(), src[perm])


@CUDA
@JIT
@BATCH_API
def test_prefill_hit_d2d_matches_sources():
    cache, sources = _make_cache()
    # layer 1 residents: two real hits in the hit region, one expert stuck in a
    # buffer slot (< 2E) whose cache bytes are poisoned -- it must come back via
    # H2D, so poison must never surface in the buffer.
    _seed_resident(cache, sources, layer_id=1, expert_id=1, slot=17)
    _seed_resident(cache, sources, layer_id=1, expert_id=3, slot=20)
    _seed_resident(cache, sources, layer_id=1, expert_id=0, slot=2)
    for name in sources:
        cache.bank_caches[name][2].fill_(float("nan"))

    cache.begin_prefill()
    assert cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    for layer_id in (0, 1):
        views = cache.wait_prefill_layer(layer_id)
        torch.cuda.synchronize()
        for view, (name, per_layer) in zip(views, sources.items()):
            assert torch.equal(view.cpu(), per_layer[layer_id]), (layer_id, name)
        cache.release_prefill_layer(layer_id)
    # prefetch(0) had no hits; prefetch(1) served experts 1 and 3 from the cache.
    assert cache.prefill_hit_rows == 2
    assert cache.prefill_total_rows == 2 * E
    # the buffer-slot resident was invalidated (its slot belongs to buffer 0)...
    assert int(cache.slot_for_id[1, 0].item()) == -1
    # ...while true hits keep their cache residency.
    assert int(cache.slot_for_id[1, 1].item()) == 17
    assert int(cache.slot_for_id[1, 3].item()) == 20


@CUDA
@JIT
@BATCH_API
@pytest.mark.parametrize("nhit", [0, E])
def test_prefill_hit_d2d_pure_extremes(nhit):
    cache, sources = _make_cache()
    for e in range(nhit):
        _seed_resident(cache, sources, layer_id=2, expert_id=e, slot=2 * E + e)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(2)
    views = cache.wait_prefill_layer(2)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[2]), name
    assert cache.prefill_hit_rows == nhit


@CUDA
def test_prefill_hit_d2d_noop_without_spare_slots():
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=2 * E,  # the cpu backend's pinned geometry: no hit region
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    cache.begin_prefill()
    assert not cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    views = cache.wait_prefill_layer(0)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[0]), name


@CUDA
@JIT
@BATCH_API
@pytest.mark.parametrize("nhit", [0, 4, 8])
def test_nvfp4_reuse_buffer_wraps_and_rebuild(nhit):
    """Exercise real large-bank gathers plus whole small-bank copies.

    The original BF16 fixtures are below the small-bank cutoff and therefore
    cannot detect broken D2D gathers. These row sizes match Qwen3.6 NVFP4.
    """
    cache = OffloadMoeCache(
        num_layers=3, num_experts=E, cache_size=5 * E,
        device=torch.device("cuda"), prefill_overlap=True,
        prefill_hit_d2d=True, quant_format="nvfp4",
    )
    sizes = (1048576, 131072, 2048, 524288, 65536, 4096)
    sources = {name: [torch.randint(0, 256, (E, size), dtype=torch.uint8).pin_memory()
                      for _ in range(3)]
               for name, size in zip(cache.bank_schema, sizes)}
    cache.set_bank_sources(sources)
    for capacity in (5 * E, 2 * E, 5 * E):
        if capacity != cache.cache_size:
            cache.rebuild(capacity)
        for chunk in range(3):
            torch.cuda.synchronize()
            cache.slot_for_id.fill_(-1)
            cache.id_of_slot.fill_(-1)
            # Rotate the expert set between chunks; every layer has independent
            # stable slots. Volatile residents are poisoned and must be refetched.
            selected = [(2 * i + i // 4 + chunk) % E for i in range(nhit)]
            for layer in range(3):
                if capacity > 2 * E:
                    for index, expert in enumerate(selected):
                        _seed_resident(cache, sources, layer, expert, 2 * E + layer * E + index)
                expert = next((i for i in range(E) if i not in selected), None)
                if expert is not None:
                    _seed_resident(cache, sources, layer, expert, layer)
                    for bank in cache.bank_caches.values():
                        bank[layer].bitwise_not_()
            cache.begin_prefill()
            assert cache._prefill_hit_d2d_active == (capacity > 2 * E)
            outputs = []
            for layer in range(3):
                cache.prefetch_prefill_layer(layer)
                cache.prefetch_prefill_layer(layer + 1)
                outputs.append([v.clone() for v in cache.wait_prefill_layer(layer)])
                cache.release_prefill_layer(layer)
            # Copies/clones above are ordered only by the production stream
            # protocol, including reuse of buffer 0 for layer 2.
            torch.cuda.synchronize()
            for layer, views in enumerate(outputs):
                for view, per_layer in zip(views, sources.values()):
                    assert torch.equal(view.cpu(), per_layer[layer])
