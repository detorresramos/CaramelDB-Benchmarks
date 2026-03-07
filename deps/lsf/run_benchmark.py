"""Run LSF benchmarks on CaramelDB-format data.

Usage:
    python deps/lsf/run_benchmark.py keys.txt values.bin [--competitor CSF|BuRR|all|LSF]

Converts data to .lrbin, optionally trains models, runs ribbon_learned_bench
natively, returns JSON.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, ".."))

from lsf.convert_to_lrbin import write_lrbin

DATASET_NAME = "caramel"
MODEL_CACHE_DIR = os.path.join(_dir, "model_cache")

_LSF_BUILD_DIR = os.path.normpath(os.path.join(_dir, "..", "LearnedStaticFunction", "build"))
_NATIVE_BENCH_BIN = os.path.join(_LSF_BUILD_DIR, "ribbon_learned_bench")


def _cache_key(keys, values):
    """Hash dataset to create a stable cache key."""
    h = hashlib.sha256()
    h.update(str(len(keys)).encode())
    h.update(keys[0].encode())
    h.update(keys[-1].encode())
    h.update(np.array(values, dtype=np.uint32).tobytes()[:4096])
    h.update(np.array(values, dtype=np.uint32).tobytes()[-4096:])
    return h.hexdigest()[:16]


def _cache_models(data_dir, dataset_name, cache_dir):
    """Copy trained models and standardized features to persistent cache."""
    os.makedirs(cache_dir, exist_ok=True)
    model_dir = os.path.join(data_dir, f"{dataset_name}_models")
    if os.path.isdir(model_dir):
        dst = os.path.join(cache_dir, f"{dataset_name}_models")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(model_dir, dst)
    # Also cache the standardized X.lrbin (features are transformed by scaler)
    x_path = os.path.join(data_dir, f"{dataset_name}_X.lrbin")
    if os.path.isfile(x_path):
        shutil.copy2(x_path, os.path.join(cache_dir, f"{dataset_name}_X.lrbin"))
    sys.stderr.write(f"Cached models to {cache_dir}\n")


def _restore_from_cache(cache_dir, data_dir, dataset_name):
    """Restore cached models and standardized features. Returns True if cache hit."""
    cached_models = os.path.join(cache_dir, f"{dataset_name}_models")
    cached_x = os.path.join(cache_dir, f"{dataset_name}_X.lrbin")
    if not os.path.isdir(cached_models):
        return False
    dst_models = os.path.join(data_dir, f"{dataset_name}_models")
    shutil.copytree(cached_models, dst_models)
    if os.path.isfile(cached_x):
        shutil.copy2(cached_x, os.path.join(data_dir, f"{dataset_name}_X.lrbin"))
    sys.stderr.write(f"Restored models from cache: {cache_dir}\n")
    return True


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
    """Train TFLite models for learned CSF."""
    train_script = os.path.join(_dir, "train_models.py")
    result = subprocess.run(
        [sys.executable, train_script, data_dir, dataset_name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(f"Model training failed:\n{result.stderr}\n")
        if result.stdout:
            sys.stderr.write(f"stdout:\n{result.stdout}\n")
        return False
    sys.stderr.write(result.stdout)
    return True


def _native_bench_available():
    return os.path.isfile(_NATIVE_BENCH_BIN)


def run_lsf_native(keys, values, competitor="all", seed=42, learned=False):
    """Run LSF benchmark natively.

    Args:
        keys: list of string keys
        values: numpy array of uint32 values
        competitor: "CSF", "BuRR", "LSF", or "all"
        seed: random seed
        learned: if True, train models locally first

    Returns list of parsed result dicts, one per competitor/storage combo.
    """
    with tempfile.TemporaryDirectory(prefix="lsf_bench_") as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)

        output_prefix = os.path.join(data_dir, DATASET_NAME)
        write_lrbin(keys, values, output_prefix)

        if learned:
            cache_dir = os.path.join(MODEL_CACHE_DIR, _cache_key(keys, values))
            if not _restore_from_cache(cache_dir, data_dir, DATASET_NAME):
                if not _train_models_locally(data_dir, DATASET_NAME):
                    return None
                _cache_models(data_dir, DATASET_NAME, cache_dir)

        env = os.environ.copy()
        env["DYLD_LIBRARY_PATH"] = _LSF_BUILD_DIR + ":" + env.get("DYLD_LIBRARY_PATH", "")

        cmd = [
            _NATIVE_BENCH_BIN,
            "-r", data_dir + "/",
            "-d", DATASET_NAME,
            "-c", competitor,
            "-s", "filter_huf",  # only the canonical Filtered-Huffman_Opt variant
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env
        )

        if result.returncode != 0:
            sys.stderr.write(f"LSF benchmark failed (rc={result.returncode}):\n{result.stderr}\n")
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
            "construction_time_s": (
                r.get("construct_ms", 0) / 1000.0
                + r.get("training_seconds", 0)
            ),
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

    if not _native_bench_available():
        print("Native benchmark binary not found. Build it first.")
        sys.exit(1)
    results = run_lsf_native(keys, values, args.competitor, learned=args.learned)
    if results:
        converted = results_to_json(results, keys, values, args.competitor)
        print(json.dumps(converted, indent=2))
    else:
        print("No results")
        sys.exit(1)


if __name__ == "__main__":
    main()
