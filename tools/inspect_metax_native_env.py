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

import importlib.util
import inspect
import os
import shutil
from pathlib import Path

import torch


def _add_root(roots, path):
    if path and path not in roots:
        roots.append(path)


def main():
    print("torch:", torch.__version__)
    print("torch_file:", Path(torch.__file__).resolve())

    spec = importlib.util.find_spec("mctlassEx")
    package_root = None
    if spec is not None:
        print("mctlassEx_spec:", spec.origin)
        if spec.submodule_search_locations:
            package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
        elif spec.origin:
            package_root = Path(spec.origin).resolve().parent
    else:
        print("mctlassEx_spec: not found")

    try:
        import mctlassEx

        print("mctlassEx_import: OK")
        print("mctlassEx_file:", Path(mctlassEx.__file__).resolve())
        package_root = Path(mctlassEx.__file__).resolve().parent
    except Exception as exc:
        print(f"mctlassEx_import: {type(exc).__name__}: {exc}")

    print("MACA_PATH:", os.environ.get("MACA_PATH"))
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
    print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
    print("CUCC_PATH:", os.environ.get("CUCC_PATH"))
    for name in (
        "mxcc",
        "mcclang",
        "mcc",
        "ninja",
        "nvcc",
        "cucc",
        "cucc++",
        "gcc",
        "g++",
        "x86_64-conda-linux-gnu-gcc",
        "x86_64-conda-linux-gnu-g++",
    ):
        print(f"{name}:", shutil.which(name))

    from torch.utils import cpp_extension

    try:
        print("GET_WCUDA_GNU_PATH_SOURCE_BEGIN")
        print(inspect.getsource(cpp_extension.get_wcuda_gnu_path))
        print("GET_WCUDA_GNU_PATH_SOURCE_END")
    except Exception as exc:
        print(f"get_wcuda_gnu_path source: {type(exc).__name__}: {exc}")

    tool_roots = (
        Path("/opt/maca/tools/cu-bridge"),
        Path("/root/cu-bridge"),
        Path("/opt/cu-bridge"),
        Path("/workspace/cu-bridge"),
    )
    for root in tool_roots:
        print(f"TOOL_ROOT {root} exists={root.exists()}")
        if not root.exists():
            continue
        children = sorted(root.iterdir())
        print("TOOL_ROOT_CHILDREN", [str(path) for path in children[:40]])
        matches = set()
        for pattern in ("**/nvcc", "**/cucc*", "**/*gcc*", "**/*g++*"):
            for path in root.glob(pattern):
                matches.add(path.resolve())
                if len(matches) >= 100:
                    break
        for path in sorted(matches):
            print("TOOL", path)

    roots = []
    for key in ("MACA_PATH", "CUDA_HOME"):
        value = os.environ.get(key)
        if value:
            _add_root(roots, Path(value) / "include")
    _add_root(roots, Path("/opt/conda/include"))
    _add_root(roots, Path(torch.__file__).resolve().parent / "include")
    _add_root(roots, package_root)

    patterns = (
        "**/mctlass_ex.h",
        "**/mma_multistage.h",
        "**/*moe*.h",
        "**/*Moe*.h",
        "**/*moe*.cuh",
        "**/*Moe*.cuh",
    )
    headers = set()
    for root in roots:
        print(f"SEARCH_ROOT {root} exists={root.exists()}")
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                headers.add(path.resolve())
    for path in sorted(headers):
        print("HEADER", path)
    print("HEADER_COUNT", len(headers))

    mctlass_roots = []
    for key in ("MACA_PATH", "CUDA_HOME"):
        value = os.environ.get(key)
        if value:
            _add_root(mctlass_roots, Path(value) / "include" / "mctlass")
    _add_root(mctlass_roots, Path("/opt/conda/include/mctlass"))
    relatives = (
        "gemm/threadblock/mma_multistage.h",
        "gemm/threadblock/default_mma_core_sm80.h",
        "gemm/warp/mma_tensor_op.h",
        "arch/mma_sm80.h",
        "mctlass_ex.h",
    )
    for root in mctlass_roots:
        print(f"MCTLASS_ROOT {root} exists={root.exists()}")
        for relative in relatives:
            path = root / relative
            print(f"  {relative}: exists={path.exists()} path={path}")


if __name__ == "__main__":
    main()
