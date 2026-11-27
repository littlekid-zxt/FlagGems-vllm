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

"""Profile vendor and FlagGems MetaX fused MoE on one benchmark shape."""

import argparse
from pathlib import Path

import torch
import triton

if torch.cuda.is_available() and "MetaX" in torch.cuda.get_device_name(0):
    from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: F401
        fused_experts_impl as _metax_fused_experts_impl,
    )

import flaggems_vllm
from benchmark import test_fused_moe as moe_bench


def _self_device_us(event):
    return getattr(
        event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0)
    )


def _profile_op(name, op, args, warmup, iters, top):
    for _ in range(warmup):
        op(*args)
    torch.cuda.synchronize()

    latency = triton.testing.do_bench(lambda: op(*args), warmup=10, rep=50)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        for _ in range(iters):
            op(*args)
        torch.cuda.synchronize()

    print(f"\n{name}: do_bench={latency:.6f} ms")
    events = [
        event
        for event in profiler.key_averages()
        if _self_device_us(event) > 0
    ]
    events.sort(key=_self_device_us, reverse=True)
    total = sum(_self_device_us(event) for event in events)
    print(f"device total={total / 1000:.6f} ms; top kernels:")
    for event in events[:top]:
        device_us = _self_device_us(event)
        print(
            f"  {device_us / 1000:.6f} ms | {100 * device_us / total:.1f}% | "
            f"calls={event.count} | {event.key}"
        )
    return latency


def _capture_jit_kernels():
    captured = []
    original_run = triton.runtime.jit.JITFunction.run

    def capturing_run(self, *args, **kwargs):
        kernel = original_run(self, *args, **kwargs)
        name = getattr(self, "__name__", getattr(self, "_fn_name", ""))
        if name in (
            "fused_moe_kernel",
            "_metax_pair_fused_moe_kernel",
            "_metax_persistent_fused_moe_kernel",
        ):
            captured.append(kernel)
        return kernel

    return captured, original_run, capturing_run


def _dump_compiled_kernels(kernels, index, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for kernel in kernels:
        kernel_key = getattr(kernel, "hash", id(kernel))
        if kernel_key in seen:
            continue
        seen.add(kernel_key)
        kernel_id = len(seen)
        kernel._init_handles()
        metadata = kernel.metadata
        print(
            f"IR kernel={kernel.name} id={kernel_id} shared={metadata.shared} "
            f"warps={metadata.num_warps} stages={metadata.num_stages} "
            f"pipeline={getattr(metadata, 'pipeline', '')} "
            f"pipeline_load_num={getattr(metadata, 'pipeline_load_num', '')} "
            f"regs={getattr(kernel, 'n_regs', '?')} "
            f"spills={getattr(kernel, 'n_spills', '?')}"
        )
        safe_name = "".join(
            char if char.isalnum() or char in "._-" else "_" for char in kernel.name
        )
        for kind, asm in kernel.asm.items():
            if not isinstance(asm, str):
                continue
            output_path = output_dir / f"index{index}_{safe_name}_{kernel_id}.{kind}"
            output_path.write_text(asm)
            if kind in ("ttir", "ttgir"):
                counts = {
                    token: asm.count(token)
                    for token in (
                        "tt.dot",
                        "ttg.dot",
                        "ttg.convert_layout",
                        "tt.load",
                        "ttg.local_load",
                        "ttg.local_alloc",
                    )
                }
                print(f"  {output_path.name}: {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1, help="index in benchmark _SHAPES")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--dump-ir", type=Path, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for index, shape in enumerate(moe_bench._SHAPES):
            print(index, shape)
        return

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shape = moe_bench._SHAPES[args.index]
    inputs = next(moe_bench._input_fn(shape, dtype, flaggems_vllm.device))
    print("index:", args.index, "shape:", shape, "dtype:", dtype)

    baseline = moe_bench.production_moe_op("metax")
    assert baseline is not None
    captured, original_run, capturing_run = _capture_jit_kernels()
    if args.dump_ir is not None:
        triton.runtime.jit.JITFunction.run = capturing_run
    latencies = {}
    try:
        for name, op in (
            ("vendor", baseline),
            ("metax", flaggems_vllm.fused_experts_impl),
        ):
            latencies[name] = _profile_op(
                name, op, inputs, args.warmup, args.iters, args.top
            )
    finally:
        triton.runtime.jit.JITFunction.run = original_run
    if args.dump_ir is not None:
        _dump_compiled_kernels(captured, args.index, args.dump_ir)
    print(f"\nspeedup vendor/metax = {latencies['vendor'] / latencies['metax']:.4f}x")


if __name__ == "__main__":
    main()
