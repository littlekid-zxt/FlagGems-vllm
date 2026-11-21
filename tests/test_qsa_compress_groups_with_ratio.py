# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg

pytestmark = pytest.mark.qsa_compress_groups_with_ratio

try:
    if flaggems_vllm.vendor_name == "hygon":
        from vllm.models.qwen4_exp.amd.ops.qsa import (
            qsa_compress_groups_with_ratio as vendor_qsa,
        )
    elif flaggems_vllm.vendor_name == "ascend":
        from vllm_ascend.ops.triton.qwen4_exp.qsa import (
            qsa_compress_groups_with_ratio as vendor_qsa,
        )
    else:
        from vllm.models.qwen4_exp.nvidia.ops.qsa import (
            qsa_compress_groups_with_ratio as vendor_qsa,
        )
except ImportError:
    vendor_qsa = None


device = flaggems_vllm.device
_device_module = getattr(torch, device, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()
if not _HAS_DEVICE:
    _HAS_DEVICE = bool(flaggems_vllm.runtime.torch_device_fn.device_count())

DTYPES = (torch.bfloat16,)


@dataclass(frozen=True)
class QsaCompressGroupsScenario:
    name: str
    query_lens: tuple[int, ...]
    history_lens: tuple[int, ...]
    state_size: int = 4
    state_blocks: int = 2712
    use_rope: bool = True
    head_dim: int = 128
    compress_ratio: int = 4

    @property
    def rows(self):
        return sum(self.query_lens)


def _model_scenario(rows, state_blocks):
    return QsaCompressGroupsScenario(
        name=f"model_r{rows}_p{state_blocks}",
        query_lens=(rows,),
        history_lens=(0 if rows > 512 else 7,),
        state_blocks=state_blocks,
    )


# Keep the functional coverage aligned with the benchmark's representative
# model shapes without importing benchmark code into the test suite.
MODEL_SCENARIOS = tuple(
    _model_scenario(rows, state_blocks)
    for state_blocks, rows_per_pool in (
        (2712, (1, 32, 128, 512, 902, 2370, 6177, 16384)),
        (512, (512,)),
    )
    for rows in rows_per_pool
)

VLLM_SCENARIOS = tuple(
    [
        QsaCompressGroupsScenario(
            name=f"vllm_streaming_rope{int(use_rope)}",
            query_lens=(7, 6),
            history_lens=(2, 5),
            state_blocks=2,
            use_rope=use_rope,
            head_dim=8,
        )
        for use_rope in (True, False)
    ]
    + [
        QsaCompressGroupsScenario(
            name=f"vllm_mixed_s{state_size}_rope{int(use_rope)}",
            query_lens=(1, 1, 37),
            history_lens=(260, 259, 138),
            state_size=state_size,
            state_blocks=3,
            use_rope=use_rope,
        )
        for state_size in (4, 8)
        for use_rope in (True, False)
    ]
    + [
        QsaCompressGroupsScenario(
            name="vllm_fresh",
            query_lens=(37,),
            history_lens=(0,),
            state_blocks=1,
        ),
        QsaCompressGroupsScenario(
            name="vllm_tiled",
            query_lens=(4097,),
            history_lens=(0,),
            state_blocks=1,
        ),
    ]
)

SCENARIOS = MODEL_SCENARIOS + VLLM_SCENARIOS


def make_scenario_inputs(scenario, dtype, seed=20261119):
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = scenario.rows
    num_requests = len(scenario.query_lens)
    raw_keys = torch.randn(
        rows,
        1,
        scenario.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    state_cache = torch.randn(
        scenario.state_blocks,
        scenario.state_size,
        1,
        scenario.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if rows:
        logical_positions = torch.cat(
            [
                torch.arange(
                    history,
                    history + query,
                    dtype=torch.int64,
                    device=device,
                )
                for query, history in zip(
                    scenario.query_lens, scenario.history_lens, strict=True
                )
            ]
        )
        token_to_req = torch.repeat_interleave(
            torch.arange(num_requests, dtype=torch.int32, device=device),
            torch.tensor(scenario.query_lens, dtype=torch.int64, device=device),
        )
    else:
        logical_positions = torch.empty(0, dtype=torch.int64, device=device)
        token_to_req = torch.empty(0, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(scenario.query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    raw_positions = torch.stack(
        (
            logical_positions,
            logical_positions // 7 + 3,
            logical_positions // 13 + 11,
        ),
        dim=1,
    ).unsqueeze(1)
    block_table = torch.arange(
        num_requests, dtype=torch.int32, device=device
    ).unsqueeze(1)
    compressed_slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
    completed = (logical_positions + 1) % scenario.compress_ratio == 0
    compressed_slots[completed] = torch.arange(
        int(completed.sum()), dtype=torch.int64, device=device
    )
    rope_cache = None
    if scenario.use_rope:
        rope_cache = torch.arange(
            scenario.state_blocks * scenario.state_size * 3,
            dtype=torch.int64,
            device=device,
        ).reshape(scenario.state_blocks, scenario.state_size, 1, 3)
    return (
        raw_keys,
        raw_positions,
        state_cache,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        scenario.compress_ratio,
        rope_cache,
    )


def make_inputs(
    num_requests,
    rows_per_request,
    compress_ratio,
    head_dim,
    dtype,
    use_rope_cache,
    seed,
):
    scenario = QsaCompressGroupsScenario(
        name="general",
        query_lens=(rows_per_request,) * num_requests,
        history_lens=tuple(
            2 * compress_ratio - 1 + request * 4 * compress_ratio
            for request in range(num_requests)
        ),
        state_size=max(8, compress_ratio),
        state_blocks=max(1, num_requests),
        use_rope=use_rope_cache,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
    )
    return make_scenario_inputs(scenario, dtype, seed)


def torch_reference(args):
    (
        raw_keys,
        raw_positions,
        state_cache,
        block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio,
        rope_cache,
    ) = args
    rows = token_to_req.numel()
    requests = token_to_req.long()
    row_ids = torch.arange(rows, dtype=torch.int64, device=device)
    query_starts = query_start_loc[requests].long()
    chunk_starts = logical_positions - (row_ids - query_starts)
    group_offsets = torch.arange(compress_ratio, dtype=torch.int64, device=device)
    positions = (
        logical_positions[:, None] - (compress_ratio - 1 - group_offsets)[None, :]
    )
    use_raw = positions >= chunk_starts[:, None]
    raw_rows = query_starts[:, None] + positions - chunk_starts[:, None]
    safe_raw_rows = raw_rows.clamp(0, rows - 1)
    raw_values = raw_keys[safe_raw_rows, 0].float()
    state_blocks = block_table[requests, 0].long()
    state_values = state_cache[
        state_blocks[:, None], positions % state_cache.shape[1], 0
    ].float()
    values = torch.where(use_raw[:, :, None], raw_values, state_values)
    valid = (compressed_slots >= 0) & (logical_positions >= compress_ratio - 1)
    pooled = torch.where(
        valid[:, None, None],
        values.sum(dim=1)[:, None, :] / compress_ratio,
        torch.zeros_like(raw_keys, dtype=torch.float32),
    ).to(raw_keys.dtype)

    first_position = logical_positions - compress_ratio + 1
    first_from_raw = first_position >= chunk_starts
    raw_first_rows = query_starts + first_position - chunk_starts
    raw_first_positions = raw_positions[raw_first_rows.clamp(0, rows - 1), 0]
    if rope_cache is None:
        first_positions = first_position[:, None].expand(rows, 3)
    else:
        state_first_positions = rope_cache[
            state_blocks, first_position % state_cache.shape[1], 0
        ]
        first_positions = torch.where(
            first_from_raw[:, None], raw_first_positions, state_first_positions
        )
    first_positions = torch.where(
        valid[:, None], first_positions, torch.zeros_like(first_positions)
    )
    return pooled, first_positions


def assert_outputs(actual, expected):
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    tolerance = (
        {"rtol": 2e-2, "atol": 1e-2}
        if actual[0].dtype == torch.bfloat16
        else {"rtol": 0.1, "atol": 0.05}
    )
    torch.testing.assert_close(actual[0].float(), expected[0].float(), **tolerance)


def check_accuracy(num_requests, rows_per_request, ratio, head_dim, use_rope, dtype):
    args = make_inputs(
        num_requests,
        rows_per_request,
        ratio,
        head_dim,
        dtype,
        use_rope,
        seed=20261119 + ratio + head_dim,
    )
    check_inputs(args)


def check_scenario_accuracy(scenario, dtype):
    args = make_scenario_inputs(scenario, dtype)
    check_inputs(args)


def check_inputs(args):
    expected = torch_reference(args)
    actual = flaggems_vllm.qsa_compress_groups_with_ratio(*args)
    assert_outputs(actual, expected)
    if vendor_qsa is not None:
        assert_outputs(vendor_qsa(*args), expected)


_GENERAL_CASES = [
    (2, 2, 4, 16, use_rope, dtype) for use_rope in (False, True) for dtype in DTYPES
]
if not cfg.QUICK_MODE:
    _GENERAL_CASES += [
        (requests, rows, ratio, dim, use_rope, dtype)
        for dtype in DTYPES
        for requests, rows, ratio, dim, use_rope in (
            (2, 2, 2, 16, True),
            (2, 2, 8, 16, True),
            (4, 4, 4, 48, False),
            (2, 3, 4, 48, True),
        )
    ]
_SCENARIO_CASES = [
    (scenario, dtype)
    for scenario in (SCENARIOS[:1] if cfg.QUICK_MODE else SCENARIOS)
    for dtype in DTYPES
]


@pytest.mark.parametrize(
    "num_requests,rows_per_request,ratio,head_dim,use_rope,dtype",
    _GENERAL_CASES,
)
@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_compress_groups_with_ratio_accuracy(
    num_requests, rows_per_request, ratio, head_dim, use_rope, dtype
):
    check_accuracy(num_requests, rows_per_request, ratio, head_dim, use_rope, dtype)


@pytest.mark.parametrize(
    "scenario,dtype",
    _SCENARIO_CASES,
    ids=lambda value: (
        value.name if isinstance(value, QsaCompressGroupsScenario) else None
    ),
)
@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_compress_groups_with_ratio_benchmark_scenarios(scenario, dtype):
    check_scenario_accuracy(scenario, dtype)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_compress_groups_with_ratio_empty():
    args = make_inputs(0, 0, 4, 16, torch.bfloat16, False, 0)
    with pytest.raises(ValueError, match="terminal offset"):
        flaggems_vllm.qsa_compress_groups_with_ratio(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_compress_groups_with_ratio_unsupported_dtype():
    args = list(make_inputs(2, 2, 4, 16, torch.bfloat16, False, 0))
    args[0] = args[0].to(torch.float16)
    args[2] = args[2].to(torch.float16)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.qsa_compress_groups_with_ratio(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_compress_groups_with_ratio_invalid_ratio():
    args = list(make_inputs(2, 2, 4, 16, torch.bfloat16, False, 0))
    args[8] = 0
    with pytest.raises(ValueError):
        flaggems_vllm.qsa_compress_groups_with_ratio(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_compress_groups_with_ratio_invalid_positions_dtype():
    args = list(make_inputs(2, 2, 4, 16, torch.bfloat16, False, 0))
    args[1] = args[1].to(torch.int32)
    with pytest.raises(ValueError):
        flaggems_vllm.qsa_compress_groups_with_ratio(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_compress_groups_with_ratio_invalid_query_starts():
    args = list(make_inputs(2, 2, 4, 16, torch.bfloat16, False, 0))
    args[5] = args[5][:1]
    with pytest.raises(ValueError):
        flaggems_vllm.qsa_compress_groups_with_ratio(*args)


@pytest.mark.skipif(not _HAS_DEVICE, reason="requires an available accelerator")
def test_qsa_compress_groups_with_ratio_invalid_state_cache():
    args = list(make_inputs(2, 2, 4, 16, torch.bfloat16, False, 0))
    args[2] = args[2][:0]
    with pytest.raises(ValueError):
        flaggems_vllm.qsa_compress_groups_with_ratio(*args)
