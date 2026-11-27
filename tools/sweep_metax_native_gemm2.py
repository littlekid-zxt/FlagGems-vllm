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
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROUND_PATTERN = re.compile(
    r"round=(\d+) native=([0-9.]+) ms vendor=([0-9.]+) ms "
    r"speedup=([0-9.]+)x"
)
REL_L2_PATTERN = re.compile(r"rel_l2=([0-9.eE+-]+)")


def _parse_output(output):
    shape = {}
    rounds = []
    rel_l2 = None
    for line in output.splitlines():
        if line.startswith("shape: "):
            shape = ast.literal_eval(line[len("shape: ") :])
        match = ROUND_PATTERN.search(line)
        if match:
            rounds.append(
                {
                    "round": int(match.group(1)),
                    "native_ms": float(match.group(2)),
                    "vendor_ms": float(match.group(3)),
                    "speedup": float(match.group(4)),
                }
            )
        match = REL_L2_PATTERN.search(line)
        if match:
            rel_l2 = float(match.group(1))
    return shape, rounds, rel_l2


def _summary(results):
    best = {}
    for result in results:
        if result["returncode"] != 0 or not result["rounds"]:
            continue
        fastest = min(result["rounds"], key=lambda item: item["native_ms"])
        key = (result["intermediate"], result["tokens"])
        candidate = {
            "grid_x": result["grid_x"],
            "native_ms": fastest["native_ms"],
            "vendor_ms": fastest["vendor_ms"],
            "speedup": fastest["speedup"],
        }
        if key not in best or candidate["native_ms"] < best[key]["native_ms"]:
            best[key] = candidate

    lines = [
        "",
        "NATIVE-GEMM-ONLY PRELIMINARY BEST (minimum native latency per shape)",
        "I       M       grid_x  native_ms  vendor_ms  speedup",
    ]
    for intermediate, tokens in sorted(best):
        item = best[(intermediate, tokens)]
        lines.append(
            f"{intermediate:<7d} {tokens:<7d} {item['grid_x']:<7d} "
            f"{item['native_ms']:<10.6f} {item['vendor_ms']:<10.6f} "
            f"{item['speedup']:.4f}x"
        )
    if best:
        mean_speedup = sum(item["speedup"] for item in best.values()) / len(best)
        lines.append(f"covered_shapes: {len(best)}/10")
        lines.append(f"mean_best_speedup_over_covered_shapes: {mean_speedup:.4f}x")
    else:
        lines.append("no successful benchmark results")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gemm1", "gemm2"), default="gemm2")
    parser.add_argument(
        "--intermediate", choices=("128", "512", "both"), default="both"
    )
    parser.add_argument(
        "--tokens", default="1036,4435,4108,13422,16384"
    )
    parser.add_argument("--grid-x", default="1,2,4,8,16,32")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--log", default="native_gemm2_sweep.log")
    parser.add_argument("--jsonl", default="native_gemm2_sweep.jsonl")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    benchmark = repo_root / "tools" / f"bench_metax_native_{args.stage}.py"
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = repo_root / log_path
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = repo_root / jsonl_path
    tokens_list = [int(value) for value in args.tokens.split(",")]
    grid_values = [int(value) for value in args.grid_x.split(",")]
    intermediates = (
        (128, 512)
        if args.intermediate == "both"
        else (int(args.intermediate),)
    )
    if args.rounds < 1:
        raise ValueError("rounds must be at least 1")

    env = os.environ.copy()
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    results = []
    log_path.write_text(
        f"native {args.stage} sweep started {datetime.now().isoformat()}\n"
        f"This is {args.stage}-only, not the full fused MoE operator.\n"
    )
    jsonl_path.write_text("")
    print("FULL_LOG", log_path)
    print("JSONL", jsonl_path)

    with log_path.open("a") as log_file, jsonl_path.open("a") as jsonl_file:
        for intermediate in intermediates:
            for tokens in tokens_list:
                for grid_x in grid_values:
                    command = [
                        sys.executable,
                        str(benchmark),
                        "--tokens",
                        str(tokens),
                        "--intermediate",
                        str(intermediate),
                        "--variant",
                        "full",
                        "--grid-x",
                        str(grid_x),
                        "--rounds",
                        str(args.rounds),
                    ]
                    label = f"I={intermediate} M={tokens} grid_x={grid_x}"
                    print(f"RUN {label}", flush=True)
                    log_file.write(f"\n===== {label} =====\n")
                    log_file.write("COMMAND " + " ".join(command) + "\n")
                    log_file.flush()
                    completed = subprocess.run(
                        command,
                        cwd=repo_root,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    output = completed.stdout
                    log_file.write(output)
                    log_file.flush()
                    shape, rounds, rel_l2 = _parse_output(output)
                    result = {
                        "intermediate": intermediate,
                        "tokens": tokens,
                        "grid_x": grid_x,
                        "returncode": completed.returncode,
                        "shape": shape,
                        "rounds": rounds,
                        "rel_l2": rel_l2,
                    }
                    results.append(result)
                    jsonl_file.write(json.dumps(result) + "\n")
                    jsonl_file.flush()
                    if completed.returncode != 0:
                        tail = output.strip().splitlines()[-1:]
                        print(f"FAIL {label} {tail}", flush=True)
                    elif rounds:
                        fastest = min(rounds, key=lambda item: item["native_ms"])
                        print(
                            f"PASS {label} native={fastest['native_ms']:.6f} ms "
                            f"vendor={fastest['vendor_ms']:.6f} ms "
                            f"speedup={fastest['speedup']:.4f}x "
                            f"rel_l2={rel_l2:.3e}",
                            flush=True,
                        )
                    else:
                        print(f"FAIL {label} no parsed benchmark result", flush=True)

        summary = _summary(results)
        for line in summary:
            print(line)
            log_file.write(line + "\n")
        log_file.write(f"finished {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
