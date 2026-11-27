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

import functools
import os
from pathlib import Path

import torch


_GEMM1_GRIDS = {
    (128, 1036): 1,
    (128, 4435): 4,
    (128, 4108): 1,
    (128, 13422): 2,
    (128, 16384): 16,
    (512, 1036): 1,
    (512, 4435): 1,
    (512, 4108): 1,
    (512, 13422): 4,
    (512, 16384): 32,
}
_GEMM2_GRIDS = {
    (128, 1036): 8,
    (128, 4435): 2,
    (128, 4108): 2,
    (128, 13422): 4,
    (128, 16384): 4,
    (512, 1036): 2,
    (512, 4435): 32,
    (512, 4108): 32,
    (512, 13422): 16,
    (512, 16384): 32,
}
_AUTO_GEMM1 = {
    (128, 1036),
    (128, 4435),
    (128, 4108),
    (512, 1036),
    (512, 4435),
    (512, 4108),
}
_AUTO_GEMM2 = {
    (128, 1036),
    (128, 4435),
    (128, 4108),
    (128, 13422),
    (128, 16384),
    (512, 1036),
    (512, 4435),
    (512, 4108),
}


def get_plan(num_tokens, intermediate_size):
    mode = os.getenv("FLAGGEMS_METAX_NATIVE_MOE_MODE", "off")
    if mode not in ("off", "gemm1", "gemm2", "both", "auto"):
        raise ValueError(f"invalid FLAGGEMS_METAX_NATIVE_MOE_MODE={mode}")
    key = (intermediate_size, num_tokens)
    gemm1 = mode in ("gemm1", "both") or (mode == "auto" and key in _AUTO_GEMM1)
    gemm2 = mode in ("gemm2", "both") or (mode == "auto" and key in _AUTO_GEMM2)
    return {
        "gemm1": gemm1,
        "gemm2": gemm2,
        "gemm1_grid": _GEMM1_GRIDS.get(key, 1),
        "gemm2_grid": _GEMM2_GRIDS.get(key, 1),
    }


@functools.lru_cache(maxsize=1)
def _extension():
    maca_path = Path(os.environ.get("MACA_PATH", "/opt/maca"))
    cuda_path = maca_path / "tools" / "cu-bridge"
    if not (cuda_path / "bin" / "gnu").exists():
        raise RuntimeError(f"cu-bridge bin/gnu not found under {cuda_path}")
    os.environ["MACA_PATH"] = str(maca_path)
    os.environ["CUDA_PATH"] = str(cuda_path)
    os.environ["CUDA_HOME"] = str(cuda_path)
    os.environ["CUCC_PATH"] = str(cuda_path)
    os.environ["PATH"] = os.pathsep.join(
        [
            str(cuda_path / "bin"),
            str(maca_path / "mxgpu_llvm" / "bin"),
            str(maca_path / "bin"),
            str(cuda_path / "tools"),
            os.environ.get("PATH", ""),
        ]
    )
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(maca_path / "lib"),
            str(maca_path / "mxgpu_llvm" / "lib"),
            str(cuda_path / "lib"),
            os.environ.get("LD_LIBRARY_PATH", ""),
        ]
    )
    os.environ.setdefault("MAX_JOBS", "4")
    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = str(cuda_path)
    source = Path(__file__).resolve().parents[1] / "native" / "qwen_bf16_moe.cu"
    return cpp_extension.load(
        name="flaggems_vllm_metax_qwen_moe_native",
        sources=[str(source)],
        extra_include_paths=[str(maca_path / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        verbose=False,
    )


@functools.lru_cache(maxsize=16)
def _ones(num_valid_tokens, device):
    return torch.ones(num_valid_tokens, dtype=torch.float32, device=device)


def launch(
    a,
    b,
    c,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    topk,
    grid_x,
):
    weights = (
        topk_weights
        if topk == 1
        else _ones(a.size(0) * topk, a.device)
    )
    _extension().qwen_bf16_gemm2(
        a,
        b,
        c,
        weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        grid_x,
        topk,
    )
