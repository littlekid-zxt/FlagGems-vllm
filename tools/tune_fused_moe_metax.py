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

"""Sweep pure-Triton MetaX fused MoE stage configs with correctness checks."""

import argparse
import math

import torch
import triton

if torch.cuda.is_available() and "MetaX" in torch.cuda.get_device_name(0):
    from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: F401
        fused_experts_impl as _metax_fused_experts_impl,
    )

import flaggems_vllm
from benchmark import test_fused_moe as moe_bench
from flaggems_vllm.runtime.backend._metax.ops import fused_moe as impl


def _candidates(stage):
    stage_names = ("gemm1", "gemm2") if stage == "both" else (stage,)
    result = [("current", {})]
    if stage in ("both", "gemm1"):
        result.append(("no_pair", {"gemm1": {"PAIR_GATE_UP_DOT": False}}))
        result.append(("pair", {"gemm1": {"PAIR_GATE_UP_DOT": True}}))
        result.append((
            "unfuse_silu",
            {"gemm1": {"UNFUSE_SILU": True, "PAIR_GATE_UP_DOT": False}},
        ))
        result.append((
            "pair_bk128",
            {
                "gemm1": {
                    "PAIR_GATE_UP_DOT": True,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 128,
                    "num_warps": 4,
                    "num_stages": 1,
                }
            },
        ))
        for num_stages in (1, 2):
            result.append((
                f"pair_small_s{num_stages}",
                {
                    "gemm1": {
                        "PAIR_GATE_UP_DOT": True,
                        "BLOCK_SIZE_N": 64,
                        "BLOCK_SIZE_K": 32,
                        "num_warps": 8,
                        "num_stages": num_stages,
                    }
                },
            ))
    for pipeline in ("cpasync", "cpasync-mixed"):
        result.append((
            f"pipeline_{pipeline}",
            {name: {"pipeline": pipeline} for name in stage_names},
        ))
    result.append((
        "single_stage",
        {name: {"num_stages": 1} for name in stage_names},
    ))
    result.append((
        "pipeline_load1",
        {name: {"pipeline_load_num": 1} for name in stage_names},
    ))
    for scenario in ("reduceSmemUsage", "fullstage", "roll"):
        result.append((
            f"scenario_{scenario}",
            {name: {"scenario": scenario} for name in stage_names},
        ))
    for group_m in (2, 4, 8):
        result.append((
            f"group_m{group_m}",
            {name: {"GROUP_SIZE_M": group_m} for name in stage_names},
        ))
    if stage in ("both", "gemm2"):
        result.append(("no_defer_router_weight", {}))
        result.append(("no_swap", {"gemm2": {"SWAP_AB": False}}))
        result.append(("persistent", {"gemm2": {"PERSISTENT_GEMM": True}}))
        result.append((
            "persistent_group4",
            {"gemm2": {"PERSISTENT_GEMM": True, "GROUP_SIZE_M": 4}},
        ))
    if stage == "both":
        result.append((
            "no_pair_no_swap",
            {
                "gemm1": {"PAIR_GATE_UP_DOT": False},
                "gemm2": {"SWAP_AB": False},
            },
        ))
    result.append((
        "warp_specialize",
        {name: {"WARP_SPECIALIZE": True} for name in stage_names},
    ))
    for block_m, block_n, block_k, num_warps, num_stages in (
        (16, 64, 64, 4, 1),
        (32, 64, 64, 4, 1),
        (32, 128, 64, 4, 1),
        (32, 128, 64, 4, 2),
        (64, 64, 32, 4, 1),
        (64, 64, 64, 4, 2),
        (64, 128, 32, 8, 1),
        (64, 128, 64, 8, 2),
        (128, 64, 32, 8, 1),
        (128, 128, 32, 8, 2),
        (128, 128, 64, 8, 2),
        (128, 256, 32, 8, 2),
        (128, 128, 128, 8, 1),
    ):
        overrides = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        if stage == "both":
            stage_overrides = {name: overrides for name in stage_names}
        else:
            # The sorted-token padding is shared by both GEMMs, so their M tiles
            # must match even when only one stage is being tuned.
            other_stage = "gemm2" if stage == "gemm1" else "gemm1"
            stage_overrides = {
                stage: overrides,
                other_stage: {"BLOCK_SIZE_M": block_m},
            }
        result.append((
            f"bm{block_m}_bn{block_n}_bk{block_k}_w{num_warps}_s{num_stages}",
            stage_overrides,
        ))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--stage", choices=("both", "gemm1", "gemm2"), default="both"
    )
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--debug-numerics", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    candidates = _candidates(args.stage)
    if args.list:
        for name, _ in candidates:
            print(name)
        return
    if args.candidate is not None:
        requested = set(args.candidate)
        unknown = requested.difference(name for name, _ in candidates)
        if unknown:
            raise ValueError(f"unknown candidates: {sorted(unknown)}")
        candidates = [item for item in candidates if item[0] in requested]

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shape = moe_bench._SHAPES[args.index]
    inputs = next(moe_bench._input_fn(shape, dtype, flaggems_vllm.device))
    baseline = moe_bench.production_moe_op("metax")
    reference = baseline(*inputs).clone()
    baseline_ms = triton.testing.do_bench(lambda: baseline(*inputs), warmup=10, rep=50)
    print("index:", args.index, "shape:", shape, "dtype:", dtype)
    print("tuned stage:", args.stage)
    print(f"vendor baseline: {baseline_ms:.6f} ms")

    original = impl.try_get_optimal_moe_config
    original_dispatch = impl.dispatch_fused_moe_kernel
    original_defer_router_weight = impl._should_defer_router_weight

    def checked_dispatch(*call_args, **call_kwargs):
        result = original_dispatch(*call_args, **call_kwargs)
        output = call_args[2]
        nonfinite = int((~torch.isfinite(output)).sum())
        finite_max = float(
            torch.where(torch.isfinite(output), output.abs(), 0).max()
        )
        print(
            f"NUMERIC stage={'gemm1' if call_kwargs.get('FUSE_SILU') else 'gemm2'} "
            f"shape={tuple(output.shape)} nonfinite={nonfinite} "
            f"finite_max={finite_max:.6e}"
        )
        return result

    for name, overrides in candidates:
        printed_configs = set()

        def tuned(*call_args, **call_kwargs):
            result = original(*call_args, **call_kwargs)
            embedded = call_kwargs.get("return_is_embedded", False)
            config, is_embedded = result if embedded else (result, False)
            config = dict(config)
            stage = call_kwargs.get("gemm_stage", "gemm1")
            config.update(overrides.get(stage, {}))
            fast_path = call_kwargs.get("enable_gemm_fast_path", False)
            config_key = (stage, fast_path)
            if args.print_config and config_key not in printed_configs:
                print(
                    f"CONFIG candidate={name} stage={stage} "
                    f"fast_path={fast_path} embedded={is_embedded} config={config}"
                )
                printed_configs.add(config_key)
            return (config, is_embedded) if embedded else config

        impl.try_get_optimal_moe_config = tuned
        try:
            if name == "no_defer_router_weight":
                impl._should_defer_router_weight = lambda *args, **kwargs: False
            else:
                impl._should_defer_router_weight = original_defer_router_weight
            if args.debug_numerics:
                impl.dispatch_fused_moe_kernel = checked_dispatch
            actual = flaggems_vllm.fused_experts_impl(*inputs)
            impl.dispatch_fused_moe_kernel = original_dispatch
            ref_max = float(reference.abs().max())
            rel_l2 = float(
                (actual - reference).float().norm()
                / reference.float().norm().clamp_min(1e-12)
            )
            max_abs = float((actual - reference).abs().max())
            actual_nonfinite = int((~torch.isfinite(actual)).sum())
            reference_nonfinite = int((~torch.isfinite(reference)).sum())
            print(
                f"{name}: rel_l2={rel_l2:.3e}, "
                f"max_abs={max_abs:.3e}, ref_max={ref_max:.3e}, "
                f"actual_nonfinite={actual_nonfinite}, "
                f"reference_nonfinite={reference_nonfinite}"
            )
            if (
                actual_nonfinite
                or reference_nonfinite
                or not math.isfinite(rel_l2)
                or not math.isfinite(max_abs)
                or rel_l2 > 0.05
            ):
                raise AssertionError(
                    f"invalid numerical result: rel_l2={rel_l2}, "
                    f"max_abs={max_abs}, actual_nonfinite={actual_nonfinite}, "
                    f"reference_nonfinite={reference_nonfinite}"
                )
            latency = triton.testing.do_bench(
                lambda: flaggems_vllm.fused_experts_impl(*inputs),
                warmup=10,
                rep=50,
            )
            print(
                f"PASS {name}: {latency:.6f} ms; "
                f"speedup={baseline_ms / latency:.4f}x"
            )
        except Exception as exc:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        finally:
            impl.try_get_optimal_moe_config = original
            impl.dispatch_fused_moe_kernel = original_dispatch
            impl._should_defer_router_weight = original_defer_router_weight


if __name__ == "__main__":
    main()
