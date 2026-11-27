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
import math
import os
from pathlib import Path

import torch
import triton

MACA_PATH = Path(os.environ.get("MACA_PATH", "/opt/maca"))
CUDA_CANDIDATES = [
    Path(os.environ.get("CUDA_PATH", str(Path.home() / "cu-bridge" / "CUDA_DIR"))),
    MACA_PATH / "tools" / "cu-bridge",
    Path.home() / "cu-bridge" / "CUDA_DIR",
]
CUDA_PATH = next(
    (path for path in CUDA_CANDIDATES if (path / "bin" / "gnu").exists()),
    None,
)
if CUDA_PATH is None:
    searched = ", ".join(str(path / "bin" / "gnu") for path in CUDA_CANDIDATES)
    raise RuntimeError(f"cu-bridge bin/gnu not found; searched: {searched}")
CUCC_PATH = Path(os.environ.get("CUCC_PATH", str(CUDA_PATH)))
os.environ["MACA_PATH"] = str(MACA_PATH)
os.environ["CUDA_PATH"] = str(CUDA_PATH)
os.environ["CUDA_HOME"] = str(CUDA_PATH)
os.environ["CUCC_PATH"] = str(CUCC_PATH)
os.environ["CUCC_CMAKE_ENTRY"] = "2"
os.environ["PATH"] = os.pathsep.join(
    [
        str(CUDA_PATH / "bin"),
        str(MACA_PATH / "mxgpu_llvm" / "bin"),
        str(MACA_PATH / "bin"),
        str(CUCC_PATH / "tools"),
        str(CUCC_PATH / "bin"),
        os.environ.get("PATH", ""),
    ]
)
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
    [
        str(MACA_PATH / "lib"),
        str(MACA_PATH / "mxgpu_llvm" / "lib"),
        str(CUDA_PATH / "lib"),
        str(CUDA_PATH / "lib64"),
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
)
os.environ.setdefault("MAX_JOBS", "4")

from torch.utils.cpp_extension import load  # noqa: E402
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402,F401
    fused_experts_impl as _metax_fused_experts_impl,
)

import flaggems_vllm  # noqa: E402,F401
from flaggems_vllm.runtime.backend._metax.ops.fused_moe import (  # noqa: E402
    moe_align_block_size_no_tle,
)
from mctlassEx import FusedMoeGEMM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4435)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--variant", choices=("direct", "full"), default="direct")
    parser.add_argument("--grid-x", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    print("MACA_PATH:", MACA_PATH)
    print("CUDA_PATH:", CUDA_PATH)
    print("CUCC_PATH:", CUCC_PATH)
    source = Path(__file__).resolve().parent / "native_metax" / "qwen_bf16_gemm2.cu"
    extension = load(
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
    n = hidden_size
    k = args.intermediate
    num_valid_tokens = args.tokens * topk
    device = "cuda"
    a = torch.randn(
        (num_valid_tokens, k), dtype=torch.bfloat16, device=device
    )
    b = torch.randn(
        (num_experts, n, k), dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (args.tokens, topk),
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(
        (args.tokens, topk), dtype=torch.float32, device=device
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size_no_tle(topk_ids, 128, num_experts, None)
    )
    weights = topk_weights.view(-1)
    native_c = torch.empty(
        (num_valid_tokens, 1, n), dtype=torch.bfloat16, device=device
    )
    vendor_c = torch.empty_like(native_c)
    vendor = FusedMoeGEMM()
    vendor_block_m = vendor.get_kernel_m(
        a, b, vendor_c, num_experts, num_valid_tokens, n, k, 1
    )
    if vendor_block_m <= 0:
        raise RuntimeError(f"vendor returned invalid BLOCK_SIZE_M={vendor_block_m}")
    (
        vendor_sorted_token_ids,
        vendor_expert_ids,
        vendor_num_tokens_post_padded,
    ) = moe_align_block_size_no_tle(
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
            args.variant == "full",
            args.grid_x,
            1,
        )

    def run_vendor():
        vendor(
            num_valid_tokens,
            n,
            k,
            num_experts,
            vendor_sorted_token_ids.numel(),
            1,
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
            True,
            filter_expert=True,
        )

    run_native()
    if args.profile:
        log_dir = Path.home() / ".mctlassex"
        previous_logs = set(log_dir.glob("mctlassex_*.log"))
        os.environ["MCTLASS_LOG_ENABLE"] = "ON"
        run_vendor()
        os.environ.pop("MCTLASS_LOG_ENABLE", None)
        for log_path in set(log_dir.glob("mctlassex_*.log")) - previous_logs:
            print("MCTLASS_LOG_FILE", log_path)
            for line in log_path.read_text(errors="replace").splitlines():
                if "MCTLASS_EX_LOG" in line:
                    print("VENDOR", line)
    else:
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
            "variant": args.variant,
            "tokens": args.tokens,
            "intermediate": args.intermediate,
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
        raise AssertionError("native GEMM2 does not match the vendor reference")

    for round_index in range(args.rounds):
        native_ms = triton.testing.do_bench(run_native, warmup=20, rep=100)
        vendor_ms = triton.testing.do_bench(run_vendor, warmup=20, rep=100)
        print(
            f"round={round_index + 1} native={native_ms:.6f} ms "
            f"vendor={vendor_ms:.6f} ms speedup={vendor_ms / native_ms:.4f}x"
        )

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        for label, fn in (("native", run_native), ("vendor", run_vendor)):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as profiler:
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
            print(f"PROFILER {label}")
            print(
                profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=15
                )
            )
            for event in profiler.key_averages():
                if "mctlass::Kernel" in event.key:
                    print(f"FULL_KERNEL_NAME {label} {event.key}")
            trace_path = Path(f"metax_gemm2_{label}_trace.json")
            profiler.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text())
            for event in trace.get("traceEvents", []):
                if "mctlass::Kernel" in event.get("name", ""):
                    print(f"KERNEL_TRACE_NAME {label} {event['name']}")
                    print(f"KERNEL_TRACE_ARGS {label} {event.get('args')}")
                    break
            print("TRACE_FILE", trace_path.resolve())


if __name__ == "__main__":
    main()
