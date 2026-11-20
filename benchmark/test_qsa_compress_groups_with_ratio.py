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

from . import base

try:
    if flaggems_vllm.vendor_name == "hygon":
        from vllm.models.qwen4_exp.amd.ops.qsa import (
            qsa_compress_groups_with_ratio as vendor_qsa,
        )
    else:
        from vllm.models.qwen4_exp.nvidia.ops.qsa import (
            qsa_compress_groups_with_ratio as vendor_qsa,
        )

    _HAS_VENDOR = True
    _VENDOR_ERROR = ""
except ImportError as exc:
    vendor_qsa = None
    _HAS_VENDOR = False
    _VENDOR_ERROR = f"{type(exc).__name__}: {exc}"


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


# Representative Qwen3.8-Flash-Next-FP8-p32768d1024 qsa_prepare shapes.
# Query/history lengths are not present in that trace; these are representative
# boundary values rather than measured model metadata.
MODEL_SCENARIOS = tuple(
    _model_scenario(rows, state_blocks)
    for state_blocks, rows_per_pool in (
        (2712, (1, 32, 128, 512, 902, 2370, 6177, 16384)),
        (512, (512,)),
    )
    for rows in rows_per_pool
)

# vLLM 0.31.0 streaming, MIXED_BATCH, fresh, and tiled test layouts.
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


def make_inputs(scenario, dtype, device, seed=20261119):
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = scenario.rows
    num_requests = len(scenario.query_lens)
    raw_keys = torch.randn(
        rows,
        1,
        scenario.head_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(dtype)
    state_cache = torch.randn(
        scenario.state_blocks,
        scenario.state_size,
        1,
        scenario.head_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).to(dtype)
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


class QsaCompressGroupsWithRatioBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = list(SCENARIOS)
        self.shape_desc = (
            "Qwen3.8 model shape trace plus vLLM baseline scenarios "
            "(query_lens, history_lens, state_size, state_blocks, rope)"
        )

    def get_input_iter(self, dtype):
        for scenario in self.shapes:
            yield make_inputs(scenario, dtype, self.device)


@pytest.mark.qsa_compress_groups_with_ratio
@pytest.mark.skipif(
    not _HAS_VENDOR,
    reason=f"requires the vLLM baseline: {_VENDOR_ERROR}".rstrip(": "),
)
def test_qsa_compress_groups_with_ratio_perf_bf16():
    bench = QsaCompressGroupsWithRatioBenchmark(
        op_name="qsa_compress_groups_with_ratio",
        torch_op=vendor_qsa,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.qsa_compress_groups_with_ratio)
    bench.run()


@pytest.mark.qsa_compress_groups_with_ratio
@pytest.mark.skipif(
    not _HAS_VENDOR,
    reason=f"requires the vLLM baseline: {_VENDOR_ERROR}".rstrip(": "),
)
def test_qsa_compress_groups_with_ratio_perf_fp8():
    bench = QsaCompressGroupsWithRatioBenchmark(
        op_name="qsa_compress_groups_with_ratio",
        torch_op=vendor_qsa,
        dtypes=[torch.float8_e4m3fn],
    )
    bench.set_gems(flaggems_vllm.qsa_compress_groups_with_ratio)
    bench.run()
