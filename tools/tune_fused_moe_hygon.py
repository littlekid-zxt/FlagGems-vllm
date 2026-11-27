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

import argparse
import json
import os
from pathlib import Path

import torch
import triton

import flaggems_vllm
import flaggems_vllm.runtime.backend._hygon.ops.fused_moe as moe_impl
from benchmark.test_fused_moe import _input_fn, hygon_fused_moe_op


SHAPES = {
    "mixtral": [
        (4, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
    ],
    "deepseek": [
        (1, 256, 7168, 2048, 8),
        (4, 256, 7168, 2048, 8),
        (16, 256, 7168, 2048, 8),
        (64, 256, 7168, 2048, 8),
        (256, 256, 7168, 2048, 8),
    ],
}

CORE_VARIANTS = [
    ("default", "1", {}),
    ("unfused", "0", {}),
    ("pair", "1", {"PAIR_GATE_UP_DOT": True}),
    (
        "unfused_bk64",
        "0",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
    ),
    (
        "pair_bn64_bk64",
        "1",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "PAIR_GATE_UP_DOT": True,
        },
    ),
]

EXTENDED_VARIANTS = CORE_VARIANTS + [
    (
        "fused_bn128_bk64",
        "1",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
    ),
    (
        "unfused_bn64_bk64",
        "0",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
    ),
    (
        "unfused_bm32_bk64",
        "0",
        {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
    ),
    (
        "pair_bn128_bk32",
        "1",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "PAIR_GATE_UP_DOT": True,
        },
    ),
]


def tile(bm=16, bn=128, bk=64, warps=4, stages=2):
    return {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 1,
        "num_warps": warps,
        "num_stages": stages,
    }


FOCUSED_VARIANTS = [
    CORE_VARIANTS[0],
    CORE_VARIANTS[3],
    ("unfused_bk128_s1", "0", tile(bk=128, stages=1)),
    ("unfused_bk128_s2", "0", tile(bk=128)),
    ("unfused_bn64_bk128", "0", tile(bn=64, bk=128)),
    ("unfused_bk64_w8", "0", tile(warps=8)),
    ("unfused_bk64_s3", "0", tile(stages=3)),
    ("unfused_bn256_bk32_w8", "0", tile(bn=256, bk=32, warps=8)),
    ("unfused_bm32_bn64_bk64", "0", tile(bm=32, bn=64)),
    (
        "unfused_g1_bk128_s1",
        "0",
        {**tile(), "gemm1": {"BLOCK_SIZE_K": 128, "num_stages": 1}},
    ),
    (
        "unfused_g2_bk128_s1",
        "0",
        {**tile(), "gemm2": {"BLOCK_SIZE_K": 128, "num_stages": 1}},
    ),
]


def measure(fn, warmup, rep):
    return triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(SHAPES),
        default=list(SHAPES),
    )
    parser.add_argument(
        "--variant-set",
        choices=("core", "extended", "focused"),
        default="core",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--output",
        default="/workspace/hygon_fused_moe_tune.json",
    )
    args = parser.parse_args()

    if flaggems_vllm.vendor_name != "hygon":
        raise RuntimeError("this tuner requires the Hygon backend")
    variant_sets = {
        "core": CORE_VARIANTS,
        "extended": EXTENDED_VARIANTS,
        "focused": FOCUSED_VARIANTS,
    }
    variants = variant_sets[args.variant_set]
    original_get_config = moe_impl.try_get_optimal_moe_config
    results = []
    output = Path(args.output)

    def save_results():
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    try:
        for group in args.groups:
            for shape in SHAPES[group]:
                tensors = next(_input_fn(shape, torch.bfloat16, flaggems_vllm.device))
                hidden, w1, w2, topk_weights, topk_ids = tensors
                w1 *= shape[2] ** -0.5
                w2 *= shape[3] ** -0.5
                reference = hygon_fused_moe_op(*tensors)
                torch.cuda.synchronize()
                atol = max(1e-2, reference.abs().max().item() * 1e-5)

                for round_index in range(args.rounds):
                    baseline_ms = measure(
                        lambda: hygon_fused_moe_op(*tensors), args.warmup, args.rep
                    )
                    for variant_name, fuse_silu, overrides in variants:
                        os.environ["FLAGGEMS_HYGON_MOE_FUSE_SILU"] = fuse_silu
                        original_get_config.cache_clear()

                        def get_config_with_overrides(*fn_args, **fn_kwargs):
                            value = original_get_config(*fn_args, **fn_kwargs)
                            is_tuple = isinstance(value, tuple)
                            config, embedded = value if is_tuple else (value, None)
                            stage = fn_kwargs.get(
                                "gemm_stage",
                                fn_args[7] if len(fn_args) > 7 else "gemm1",
                            )
                            tuned = dict(config)
                            tuned.update(
                                {
                                    key: value
                                    for key, value in overrides.items()
                                    if key not in ("gemm1", "gemm2")
                                }
                            )
                            tuned.update(overrides.get(stage, {}))
                            return (tuned, embedded) if is_tuple else tuned

                        moe_impl.try_get_optimal_moe_config = get_config_with_overrides
                        result = flaggems_vllm.fused_experts_impl(*tensors)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            result, reference, rtol=1e-1, atol=atol
                        )
                        gems_ms = measure(
                            lambda: flaggems_vllm.fused_experts_impl(*tensors),
                            args.warmup,
                            args.rep,
                        )
                        speedup = baseline_ms / gems_ms
                        record = {
                            "group": group,
                            "shape": shape,
                            "round": round_index,
                            "variant": variant_name,
                            "fuse_silu": fuse_silu,
                            "overrides": overrides,
                            "baseline_ms": baseline_ms,
                            "gems_ms": gems_ms,
                            "speedup": speedup,
                        }
                        results.append(record)
                        print(
                            f"RESULT group={group} shape={shape} round={round_index} "
                            f"variant={variant_name} baseline_ms={baseline_ms:.6f} "
                            f"gems_ms={gems_ms:.6f} speedup={speedup:.4f}",
                            flush=True,
                        )
                        save_results()
    finally:
        moe_impl.try_get_optimal_moe_config = original_get_config
        original_get_config.cache_clear()
        os.environ.pop("FLAGGEMS_HYGON_MOE_FUSE_SILU", None)

    for variant_name, _, _ in variants:
        selected = [
            row["speedup"] for row in results if row["variant"] == variant_name
        ]
        if selected:
            print(
                f"SUMMARY variant={variant_name} "
                f"mean_speedup={sum(selected) / len(selected):.4f} "
                f"min_speedup={min(selected):.4f}",
                flush=True,
            )
    print(f"Saved {len(results)} results to {output}", flush=True)


if __name__ == "__main__":
    main()
