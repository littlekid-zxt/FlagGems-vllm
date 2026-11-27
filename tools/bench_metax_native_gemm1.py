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
import math
from pathlib import Path

import bench_metax_native_gemm2 as common
import torch
import triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4435)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--grid-x", type=int, default=0)
    parser.add_argument("--variant", choices=("full",), default="full")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    source = (
        Path(common.__file__).resolve().parent
        / "native_metax"
        / "qwen_bf16_gemm2.cu"
    )
    extension = common.load(
        name="flaggems_metax_native_gemm2_proto",
        sources=[str(source)],
        extra_include_paths=["/opt/maca/include"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        verbose=True,
    )
    print("native extension:", extension.__file__)
    if args.build_only:
        return
    if not torch.cuda.is_available() or "MetaX" not in torch.cuda.get_device_name(0):
        raise RuntimeError("requires a MetaX GPU")
    if args.tokens <= 0 or args.intermediate not in (128, 512):
        raise ValueError("tokens must be positive and intermediate must be 128 or 512")

    torch.manual_seed(42)
    num_experts = 256
    topk = 8
    hidden_size = 2048
    batch_size = args.tokens
    num_valid_tokens = batch_size * topk
    n = 2 * args.intermediate
    k = hidden_size
    device = "cuda"
    a = torch.randn((batch_size, k), dtype=torch.bfloat16, device=device)
    b = torch.randn(
        (num_experts, n, k), dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (batch_size, topk),
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.ones(
        (batch_size, topk), dtype=torch.float32, device=device
    )
    weights = topk_weights.view(-1)
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        common.moe_align_block_size_no_tle(
            topk_ids, 128, num_experts, None
        )
    )
    native_c = torch.empty(
        (batch_size, topk, n), dtype=torch.bfloat16, device=device
    )
    vendor_c = torch.empty_like(native_c)
    vendor = common.FusedMoeGEMM()
    vendor_block_m = vendor.get_kernel_m(
        a, b, vendor_c, num_experts, batch_size, n, k, topk
    )
    if vendor_block_m <= 0:
        raise RuntimeError(f"vendor returned invalid BLOCK_SIZE_M={vendor_block_m}")
    (
        vendor_sorted_token_ids,
        vendor_expert_ids,
        vendor_num_tokens_post_padded,
    ) = common.moe_align_block_size_no_tle(
        topk_ids, vendor_block_m, num_experts, None
    )

    def run_native():
        extension.qwen_bf16_gemm2(
            a,
            b,
            native_c,
            weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            args.grid_x,
            topk,
        )

    def run_vendor():
        vendor(
            batch_size,
            n,
            k,
            num_experts,
            vendor_sorted_token_ids.numel(),
            topk,
            a,
            b,
            vendor_c,
            None,
            None,
            None,
            weights,
            vendor_sorted_token_ids,
            vendor_expert_ids,
            vendor_num_tokens_post_padded,
            False,
            filter_expert=True,
        )

    run_native()
    run_vendor()
    torch.cuda.synchronize()
    difference = (native_c - vendor_c).float()
    rel_l2 = float(
        difference.norm() / vendor_c.float().norm().clamp_min(1e-12)
    )
    max_abs = float(difference.abs().max())
    vendor_max = float(vendor_c.abs().max())
    native_nonfinite = int((~torch.isfinite(native_c)).sum())
    vendor_nonfinite = int((~torch.isfinite(vendor_c)).sum())
    grid_m = triton.cdiv(sorted_token_ids.numel(), 128)
    group_blocks = max(
        1, min(8, triton.cdiv(num_valid_tokens // num_experts, 128))
    )
    grid_x = min(grid_m, group_blocks)
    if args.grid_x > 0:
        grid_x = min(grid_m, args.grid_x)
    native_grid = (grid_x, triton.cdiv(n, 128), triton.cdiv(grid_m, grid_x))
    print(
        "shape:",
        {
            "stage": "gemm1",
            "tokens": args.tokens,
            "intermediate": args.intermediate,
            "topk": topk,
            "native_weight": "ones",
            "vendor_mul_weight": False,
            "native_block_m": 128,
            "vendor_block_m": vendor_block_m,
            "native_EM": sorted_token_ids.numel(),
            "vendor_EM": vendor_sorted_token_ids.numel(),
            "native_grid": native_grid,
            "N": n,
            "K": k,
        },
    )
    print(
        f"correctness: rel_l2={rel_l2:.3e}, max_abs={max_abs:.3e}, "
        f"vendor_max={vendor_max:.3e}, "
        f"native_nonfinite={native_nonfinite}, "
        f"vendor_nonfinite={vendor_nonfinite}"
    )
    if (
        native_nonfinite
        or vendor_nonfinite
        or not math.isfinite(rel_l2)
        or rel_l2 > 1e-3
    ):
        raise AssertionError("native GEMM1 does not match the vendor reference")

    for round_index in range(args.rounds):
        native_ms = triton.testing.do_bench(run_native, warmup=20, rep=100)
        vendor_ms = triton.testing.do_bench(run_vendor, warmup=20, rep=100)
        print(
            f"round={round_index + 1} native={native_ms:.6f} ms "
            f"vendor={vendor_ms:.6f} ms speedup={vendor_ms / native_ms:.4f}x"
        )


if __name__ == "__main__":
    main()
