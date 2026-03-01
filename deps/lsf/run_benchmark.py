"""Run LSF benchmarks via Docker on CaramelDB-format data.

Usage:
    python deps/lsf/run_benchmark.py keys.txt values.bin [--competitor CSF|BuRR|all|LSF]

Converts data to .lrbin, optionally trains models, runs ribbon_learned_bench
in Docker, returns JSON.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, ".."))

from lsf.convert_to_lrbin import write_lrbin

DOCKER_IMAGE = "caramel-lsf"
DATASET_NAME = "caramel"


def parse_result_line(line):
    """Parse a RESULT line into a dict."""
    parts = line.strip().split()
    if not parts or parts[0] != "RESULT":
        return None
    result = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            try:
                value = float(value)
                if value == int(value):
                    value = int(value)
            except ValueError:
                pass
            result[key] = value
    return result


def _train_models_locally(data_dir, dataset_name):
    """Train TFLite models natively (outside Docker).

    Docker runs with --platform linux/amd64 via QEMU, which can't handle
    the x86 instructions in numpy/tensorflow. So we train natively and
    pass the .tflite files into Docker for benchmarking.
    """
    train_script = os.path.join(_dir, "train_models.py")
    result = subprocess.run(
        [sys.executable, train_script, data_dir, dataset_name],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        sys.stderr.write(f"Model training failed:\n{result.stderr}\n")
        if result.stdout:
            sys.stderr.write(f"stdout:\n{result.stdout}\n")
        return False
    sys.stderr.write(result.stdout)
    return True


def run_lsf_docker(keys, values, competitor="all", seed=42, learned=False):
    """Run LSF benchmark via Docker.

    Args:
        keys: list of string keys
        values: numpy array of uint32 values
        competitor: "CSF", "BuRR", "LSF", or "all"
        seed: random seed
        learned: if True, train models locally then benchmark with LSF competitor

    Returns list of parsed result dicts, one per competitor/storage combo.
    """
    with tempfile.TemporaryDirectory(prefix="lsf_bench_") as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)

        output_prefix = os.path.join(data_dir, DATASET_NAME)
        write_lrbin(keys, values, output_prefix)

        if learned:
            if not _train_models_locally(data_dir, DATASET_NAME):
                return None

        container_name = f"lsf-bench-{os.getpid()}"
        cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--platform", "linux/amd64",
            "-v", f"{data_dir}:/data",
            DOCKER_IMAGE,
            "bench-only", "/data/", DATASET_NAME, competitor,
        ]

        timeout = 3600 if learned else 600
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"LSF benchmark timed out after {timeout}s\n")
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True, timeout=30,
            )
            return None

        if result.returncode != 0:
            sys.stderr.write(f"LSF benchmark failed:\n{result.stderr}\n")
            if result.stdout:
                sys.stderr.write(f"stdout:\n{result.stdout}\n")
            return None

        results = []
        for line in result.stdout.splitlines():
            parsed = parse_result_line(line)
            if parsed:
                results.append(parsed)

        return results


def results_to_json(results, keys, values, competitor):
    """Convert parsed results to our standard JSON format.

    For learned results (comp=ours), only keeps Filtered-Huffman_Opt
    (the canonical variant from the LSF paper).
    """
    if not results:
        return None

    out = []
    for r in results:
        comp = r.get("comp", competitor)
        storage = r.get("storage_name", "unknown")

        # Only keep the canonical learned variant
        if comp == "ours" and storage != "Filtered-Huffman_Opt":
            continue

        name = f"lsf_{comp}_{storage}".lower().replace(" ", "_")

        entry = {
            "method": name,
            "construction_time_s": r.get("construct_ms", 0) / 1000.0,
            "inference_ns": {
                "mean": r.get("query_nanos", 0),
            },
            "memory": {
                "serialized": int(
                    (r.get("storage_bits", 0) + r.get("model_bits", 0))
                    * len(keys) / 8
                ),
                "bits_per_key": r.get("storage_bits", 0),
                "model_bits_per_key": r.get("model_bits", 0),
            },
            "filter_params": {
                "competitor": comp,
                "storage": storage,
                "cross_entropy_bits_per_key": r.get(
                    "cross_entropy_bit_per_key"
                ),
            },
        }
        out.append(entry)

    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("keys_path")
    parser.add_argument("values_path")
    parser.add_argument(
        "--competitor", default="all", choices=["CSF", "BuRR", "LSF", "all"]
    )
    parser.add_argument(
        "--learned", action="store_true",
        help="Train models first, then benchmark (includes LSF competitor)"
    )
    args = parser.parse_args()

    with open(args.keys_path) as f:
        keys = f.read().splitlines()
    values = np.frombuffer(
        open(args.values_path, "rb").read(), dtype=np.dtype(">u8")
    ).astype(np.uint32)

    results = run_lsf_docker(keys, values, args.competitor, learned=args.learned)
    if results:
        converted = results_to_json(results, keys, values, args.competitor)
        print(json.dumps(converted, indent=2))
    else:
        print("No results")
        sys.exit(1)


if __name__ == "__main__":
    main()
