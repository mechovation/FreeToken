"""Real graph replay: changing inputs/state slots checked against a CPU oracle."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Context, Req, get_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.graph import GraphRunner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA or HIP GPU")
def test_graph_runner_replays_changed_inputs_and_state_slots(monkeypatch):
    import freetoken.core as core

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    state = torch.zeros(5, device="cuda")
    expected = torch.zeros(5)

    class Model:
        def forward(self):
            b = get_global_ctx().batch
            slots = b.fla_metadata.cache_indices.long()
            value = state[slots] + b.input_ids.float() + b.positions.float()
            state[slots] = value
            return torch.stack((value, value * 2, b.out_loc.float()), dim=-1)

    backend = SimpleNamespace(
        init_capture_graph=lambda **kw: None,
        prepare_for_capture=lambda b: None,
        prepare_for_replay=lambda b: None,
    )
    dummy = Req(torch.tensor([0], dtype=torch.int32), 0, 0, 1, -1, None, None)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner = GraphRunner(
            stream,
            torch.device("cuda"),
            Model(),
            backend,
            [1, 2],
            2,
            1 << 30,
            16,
            3,
            dummy,
        )
        state.zero_()
        ptrs = (
            runner.buffer.input_ids.data_ptr(),
            runner.buffer.table_idx.data_ptr(),
            state.data_ptr(),
        )
        # Exercise both graphs 100 times, changing request slots on each replay.
        for step in range(200):
            bs = 1 + step % 2
            slots = torch.tensor([1 + step % 4, 1 + (step + 1) % 4])[:bs]
            ids = torch.arange(bs, dtype=torch.int32) + step % 13
            positions = torch.arange(bs, dtype=torch.int32) + step
            locations = torch.arange(bs, dtype=torch.int32) + 3 * step
            b = Batch([dummy] * bs, "decode")
            runner.pad_batch(b)
            b.input_ids, b.positions, b.out_loc = [
                t.cuda() for t in (ids, positions, locations)
            ]
            b.linear_table_idx = slots.cuda().int()
            value = expected[slots] + ids.float() + positions.float()
            expected[slots] = value
            result = runner.replay(b)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                result.cpu(),
                torch.stack((value, value * 2, locations.float()), -1),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(state.cpu(), expected, rtol=0, atol=0)
            assert ptrs == (
                runner.buffer.input_ids.data_ptr(),
                runner.buffer.table_idx.data_ptr(),
                state.data_ptr(),
            )
        runner.destroy_cuda_graphs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA or HIP GPU")
@pytest.mark.parametrize("bs", [1, 2])
def test_gdn_graph_replay_matches_recurrence_with_changing_slots(bs):
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla
    from freetoken.models.qwen3_5_moe.gdn_reference import recurrent_gated_delta_rule

    torch.manual_seed(41)
    kh, vh, dim = 2, 4, 64
    q = torch.randn(1, bs, kh, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(1, bs, vh, dim, dtype=q.dtype, device="cuda")
    a = torch.zeros(bs, vh, device="cuda")
    b = torch.zeros_like(a)
    a_log = torch.zeros(vh, device="cuda")
    bias = torch.zeros_like(a_log)
    state = torch.zeros(5, vh, dim, dim, device="cuda")
    expected_state = torch.zeros_like(state, device="cpu")
    slots = torch.arange(1, bs + 1, device="cuda", dtype=torch.int32)
    cu = torch.arange(bs + 1, device="cuda", dtype=torch.int32)

    def forward():
        return gdn_decode_fla(
            q,
            k,
            v,
            a,
            b,
            A_log=a_log,
            dt_bias=bias,
            state_source=state,
            indices=slots,
            cu_seqlens=cu,
            scale=dim**-0.5,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = forward()
    state.zero_()
    for step in range(100):
        slot_ids = (torch.arange(bs) + step) % 4 + 1
        slots.copy_(slot_ids)
        for tensor in (q, k, v, a, b):
            tensor.normal_()
        # CPU recurrence, including independent gating and GQA expansion.
        rq = q.cpu().float().transpose(0, 1).repeat_interleave(vh // kh, dim=2)
        rk = k.cpu().float().transpose(0, 1).repeat_interleave(vh // kh, dim=2)
        rv = v.cpu().float().transpose(0, 1)
        decay = -torch.nn.functional.softplus(a.cpu()).unsqueeze(1)
        beta = b.cpu().sigmoid().unsqueeze(1)
        expected, updated = recurrent_gated_delta_rule(
            rq, rk, rv, decay, beta, initial_state=expected_state[slot_ids]
        )
        expected_state[slot_ids] = updated
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual.cpu().float(), expected[:, 0], rtol=2e-2, atol=2e-2
        )
        # The vendored kernel addresses state as value-major (v * K + k),
        # whereas the mathematical recurrence returns [K, V]. The model's
        # square head dimensions hide this distinction in the tensor shape.
        torch.testing.assert_close(
            state.cpu().transpose(-1, -2), expected_state, rtol=2e-3, atol=2e-3
        )
