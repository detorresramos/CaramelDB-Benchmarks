"""Experiment runner for baseline comparisons.

Usage:
    python baselines/run_baselines.py                              # default sweep
    python baselines/run_baselines.py --alpha 0.8 --n 100000      # single config
    python baselines/run_baselines.py --filter-type binary_fuse    # specify filter
    python baselines/run_baselines.py --dataset data.csv           # custom dataset CSV
"""

import argparse
import csv
import json
import os
import sys
import time
import tracemalloc

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, ".."))

from shared.data_gen import (
    MINORITY_DISTRIBUTIONS,
    compute_actual_alpha,
    gen_alpha_values,
    gen_keys,
)
from baselines.methods import (
    CppHashTable,
    CSFFilter,
    HashTable,
    JavaCSF,
    JavaMPH,
    LSFAll,
    LSFLearned,
)

FIGURES_DIR = os.path.join(_dir, "figures")
DATA_DIR = os.path.join(FIGURES_DIR, "data")

DEFAULT_SEED = 42
DEFAULT_FILTER_TYPE = "binary_fuse"
NUM_INFERENCE_QUERIES = 100

SWEEP_NS = [100_000, 1_000_000, 10_000_000]
SWEEP_ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
SWEEP_DISTS = ["uniform_100", "zipfian", "unique"]


def measure_inference_time(method, structure, keys, seed):
    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(len(keys), size=NUM_INFERENCE_QUERIES, replace=True)
    sample_keys = [keys[i] for i in sample_indices]

    for k in sample_keys[:10]:
        method.query(structure, k)

    times_ns = []
    for k in sample_keys:
        t0 = time.perf_counter_ns()
        method.query(structure, k)
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)

    arr = np.array(times_ns, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def measure_tracemalloc(method, keys, values):
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    method.construct(keys, values)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = snap_after.compare_to(snap_before, "lineno")
    return sum(s.size_diff for s in stats if s.size_diff > 0)


def _print_result_summary(result):
    inf = result["inference_ns"]
    mem = result["memory"]
    if "tracemalloc" in mem:
        memory_desc = f"tracemalloc={mem['tracemalloc']:,}B"
    elif "serialized" in mem:
        memory_desc = f"serialized={mem['serialized']:,}B"
    else:
        memory_desc = f"serialized={mem.get('serialized_bytes', 0):,}B"
    p99_str = f" (p99={inf['p99']:.0f})" if "p99" in inf else ""
    print(
        f"    construction={result['construction_time_s']:.3f}s  "
        f"inference={inf['mean']:.0f}ns{p99_str}  "
        f"{memory_desc}"
    )


def run_java_method(method, keys, values, seed):
    result = method.run_full_benchmark(keys, values, seed)
    if result is None:
        return None
    result["filter_params"] = method.get_params()
    return result


def run_single_method(method, keys, values, seed):
    t0 = time.perf_counter()
    structure = method.construct(keys, values)
    construction_time = time.perf_counter() - t0

    memory = method.measure_memory(keys, values)
    if memory is None and hasattr(method, "measure_memory_from_structure"):
        memory = method.measure_memory_from_structure(structure)
    memory["tracemalloc"] = measure_tracemalloc(method, keys, values)

    inference_ns = measure_inference_time(method, structure, keys, seed)

    return {
        "method": method.name,
        "construction_time_s": construction_time,
        "inference_ns": inference_ns,
        "memory": memory,
        "filter_params": method.get_params(),
    }


def load_dataset(path):
    keys = []
    values = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.append(row["key"])
            values.append(int(row["value"]))
    return keys, np.array(values, dtype=np.uint32)


def run_experiment(n, alpha, minority_dist, seed, filter_type, keys=None, values=None):
    if keys is None:
        keys = gen_keys(n, seed=seed)
        values = gen_alpha_values(n, alpha, seed=seed, minority_dist=minority_dist)
    actual_alpha = compute_actual_alpha(values)

    methods = [
        HashTable(),
        CppHashTable(),
        CSFFilter(filter_type=filter_type, epsilon_strategy="optimal"),
        CSFFilter(filter_type=filter_type, epsilon_strategy="shibuya"),
    ]

    results = []
    for method in methods:
        print(f"  Running {method.name}...")
        result = run_single_method(method, keys, values, seed)
        result["filter_type"] = getattr(method, "filter_type", None)
        results.append(result)
        _print_result_summary(result)

    for java_method in [JavaCSF(), JavaMPH()]:
        print(f"  Running {java_method.name}...")
        result = run_java_method(java_method, keys, values, seed)
        if result is None:
            print("    Skipped (Java not available or JAR not built)")
            continue
        result["filter_type"] = None
        results.append(result)
        _print_result_summary(result)

    lsf_method = LSFAll()
    print(f"  Running {lsf_method.name} (Docker)...")
    lsf_results = lsf_method.run_full_benchmark(keys, values, seed)
    if lsf_results is None:
        print("    Skipped (Docker not available or image not built)")
    else:
        for r in lsf_results:
            r["filter_type"] = None
            results.append(r)
            _print_result_summary(r)

    lsf_learned = LSFLearned()
    num_classes = len(np.unique(values))
    if num_classes <= 500:
        est = "~2-5min"
    elif num_classes <= 5000:
        est = "~10-30min"
    elif num_classes <= 20000:
        est = "~30-60min, may timeout"
    else:
        est = "likely timeout under QEMU"
    print(f"  Running {lsf_learned.name} ({num_classes} classes, est {est})...", flush=True)
    learned_t0 = time.perf_counter()
    lsf_learned_results = lsf_learned.run_full_benchmark(keys, values, seed)
    learned_elapsed = time.perf_counter() - learned_t0
    if lsf_learned_results is None:
        print(f"    Skipped after {learned_elapsed:.0f}s (Docker not available or training failed)", flush=True)
    else:
        print(f"    Learned LSF completed in {learned_elapsed:.0f}s", flush=True)
        for r in lsf_learned_results:
            r["filter_type"] = None
            results.append(r)
            _print_result_summary(r)

    return {
        "dataset": {
            "N": len(keys),
            "alpha": round(actual_alpha, 4),
            "target_alpha": alpha,
            "minority_dist": minority_dist,
            "seed": seed,
        },
        "results": results,
    }


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline comparisons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=None,
        help="Number of keys (one or more). Omit for default sweep.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help="Alpha values (one or more). Omit for default sweep.",
    )
    parser.add_argument(
        "--minority-dist",
        choices=MINORITY_DISTRIBUTIONS,
        nargs="+",
        default=None,
        help="Minority distributions (one or more). Omit for default sweep.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to CSV file with 'key' and 'value' columns",
    )
    parser.add_argument(
        "--filter-type",
        choices=["xor", "binary_fuse", "bloom"],
        default=DEFAULT_FILTER_TYPE,
        help="Filter type for CSF methods",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.dataset is not None:
        keys, values = load_dataset(args.dataset)
        alpha = float(compute_actual_alpha(values))
        n = len(keys)
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        print(
            f"\n=== dataset={args.dataset}, n={n}, alpha={alpha:.4f}, filter={args.filter_type} ==="
        )

        data = run_experiment(
            n, alpha, "custom", args.seed, args.filter_type, keys=keys, values=values
        )
        data["dataset"]["source"] = args.dataset

        filename = f"baselines_{dataset_name}_{args.filter_type}.json"
        save_json(data, os.path.join(DATA_DIR, filename))
        return

    ns = args.n if args.n is not None else SWEEP_NS
    alphas = args.alpha if args.alpha is not None else SWEEP_ALPHAS
    dists = args.minority_dist if args.minority_dist is not None else SWEEP_DISTS

    configs = [(dist, n, alpha) for dist in dists for n in ns for alpha in alphas]
    total = len(configs)
    elapsed_times = []

    # Rough time estimates based on observed Docker/QEMU performance:
    # - uniform_100/zipfian: ~2-5min per config (few classes, fast TFLite)
    # - unique: ~5min (high alpha, few classes) to 60min timeout (low alpha, 50K classes)
    n_unique = sum(1 for d, _, _ in configs if d == "unique")
    n_other = total - n_unique
    est_min = n_other * 2 + n_unique * 5
    est_max = n_other * 5 + n_unique * 60
    print(
        f"Sweep: {total} configs ({n_other} uniform/zipfian, {n_unique} unique). "
        f"Estimated {est_min // 60}h{est_min % 60:02d}m - {est_max // 60}h{est_max % 60:02d}m total.",
        flush=True,
    )

    all_results = []
    for idx, (dist, n, alpha) in enumerate(configs, 1):
        eta_str = ""
        if elapsed_times:
            avg = sum(elapsed_times) / len(elapsed_times)
            remaining = avg * (total - idx + 1)
            if remaining >= 3600:
                eta_str = f" | ETA: {remaining / 3600:.1f}h"
            else:
                eta_str = f" | ETA: {remaining / 60:.0f}m"

        print(
            f"\n=== [{idx}/{total}] alpha={alpha}, dist={dist}, n={n:,}, "
            f"filter={args.filter_type}{eta_str} ===",
            flush=True,
        )

        config_t0 = time.perf_counter()
        data = run_experiment(n, alpha, dist, args.seed, args.filter_type)
        config_elapsed = time.perf_counter() - config_t0
        elapsed_times.append(config_elapsed)

        if config_elapsed >= 60:
            print(f"  Config took {config_elapsed / 60:.1f}m", flush=True)
        else:
            print(f"  Config took {config_elapsed:.1f}s", flush=True)

        all_results.append(data)

        filename = f"baselines_n{n}_a{alpha}_{dist}_{args.filter_type}.json"
        save_json(data, os.path.join(DATA_DIR, filename))

    combined = {
        "filter_type": args.filter_type,
        "ns": ns,
        "alphas": alphas,
        "dists": dists,
        "seed": args.seed,
        "experiments": all_results,
    }
    save_json(
        combined, os.path.join(DATA_DIR, f"baselines_sweep_{args.filter_type}.json")
    )


if __name__ == "__main__":
    main()
