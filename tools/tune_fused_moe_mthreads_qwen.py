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
from pathlib import Path

import torch
import torch_musa
import triton

import flaggems_vllm
import flaggems_vllm.runtime.backend._mthreads.ops.fused_moe as moe_impl
from benchmark.test_fused_moe import _input_fn, mthreads_fused_moe_op


SHAPES = {
    "i128": [
        (1036, 256, 2048, 128, 8),
        (4435, 256, 2048, 128, 8),
        (4108, 256, 2048, 128, 8),
    ],
    "i512": [
        (13422, 256, 2048, 512, 8),
        (16384, 256, 2048, 512, 8),
    ],
}

# (STAGES, BN, NUM_WARPS) for GEMM1 and GEMM2.
TLE_CONFIGS = {
    "dense_default": ((4, 256, 8), (2, 256, 16)),
    "dense_previous": ((3, 256, 8), (2, 256, 8)),
    "dense_g1s3_g2w16": ((3, 256, 8), (2, 256, 16)),
    "dense_g1s4_g2w8": ((4, 256, 8), (2, 256, 8)),
    "dense_g1s2_g2w16": ((2, 256, 8), (2, 256, 16)),
    "dense_g1w16_g2w16": ((4, 256, 16), (2, 256, 16)),
    "dense_narrow": ((3, 128, 4), (2, 128, 4)),
}


def measure(fn, warmup=100, rep=100):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=sorted(SHAPES), default=list(SHAPES))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", default="/data/zhouxt/mthreads_qwen_tune.json")
    args = parser.parse_args()

    if flaggems_vllm.vendor_name != "mthreads":
        raise RuntimeError("this tuner requires the MThreads backend")
    if not moe_impl._HAS_TLE:
        raise RuntimeError("this tuner requires FlagTree TLE")

    original_tle_gemm = moe_impl._tle_gemm
    results = []
    output = Path(args.output)

    def save_results():
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    try:
        for group in args.groups:
            for shape in SHAPES[group]:
                n_inter = shape[3]
                hidden_size = shape[2]
                tensors = next(_input_fn(shape, torch.bfloat16, flaggems_vllm.device))
                hidden, w1, w2, topk_weights, topk_ids = tensors
                w1 *= hidden_size**-0.5
                w2 *= n_inter**-0.5
                reference = mthreads_fused_moe_op(*tensors)
                torch.musa.synchronize()
                atol = max(1e-2, reference.abs().max().item() * 1e-5)

                variants = [("current", None)]
                if group == "i128":
                    variants.append(("nopad", None))
                variants.extend((name, config) for name, config in TLE_CONFIGS.items())

                for round_index in range(args.rounds):
                    baseline_ms = measure(lambda: mthreads_fused_moe_op(*tensors))
                    for variant_name, config in variants:
                        if variant_name == "current":
                            fn = lambda: flaggems_vllm.fused_experts_impl(*tensors)
                        elif variant_name == "nopad":
                            fn = lambda: moe_impl.nopad_fused_moe(*tensors)
                        else:
                            fn = lambda: moe_impl._dense_fused_moe(*tensors)

                        if config is not None:
                            g1, g2 = config

                            def configured_tle_gemm(
                                a_buf,
                                w,
                                out,
                                m_indices,
                                E,
                                total,
                                N,
                                K,
                                STAGES,
                                BN,
                                NUM_WARPS,
                            ):
                                stages, bn, warps = g1 if K == hidden_size else g2
                                return original_tle_gemm(
                                    a_buf,
                                    w,
                                    out,
                                    m_indices,
                                    E,
                                    total,
                                    N,
                                    K,
                                    stages,
                                    bn,
                                    warps,
                                )

                            moe_impl._tle_gemm = configured_tle_gemm
                        else:
                            moe_impl._tle_gemm = original_tle_gemm

                        result = fn()
                        torch.musa.synchronize()
                        torch.testing.assert_close(result, reference, rtol=1e-1, atol=atol)
                        gems_ms = measure(fn)
                        speedup = baseline_ms / gems_ms
                        results.append(
                            {
                                "group": group,
                                "shape": shape,
                                "round": round_index,
                                "variant": variant_name,
                                "baseline_ms": baseline_ms,
                                "gems_ms": gems_ms,
                                "speedup": speedup,
                            }
                        )
                        print(
                            f"RESULT group={group} shape={shape} round={round_index} "
                            f"variant={variant_name} baseline_ms={baseline_ms:.6f} "
                            f"gems_ms={gems_ms:.6f} speedup={speedup:.4f}",
                            flush=True,
                        )
                        save_results()
    finally:
        moe_impl._tle_gemm = original_tle_gemm

    for variant_name in [name for name, _ in variants]:
        selected = [row["speedup"] for row in results if row["variant"] == variant_name]
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
