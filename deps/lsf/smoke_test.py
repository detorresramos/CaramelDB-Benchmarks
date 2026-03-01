"""Smoke test for the learned LSF pipeline.

Generates a small dataset (1000 keys, uniform_100, alpha=0.8),
trains a model, runs the benchmark, and prints results.

Usage:
    python deps/lsf/smoke_test.py
    python deps/lsf/smoke_test.py --native   # skip Docker, use native binary
"""

import os
import sys
import time

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, "../.."))

from shared.data_gen import gen_alpha_values, gen_keys

from lsf.run_benchmark import results_to_json, run_lsf_docker


def main():
    native = "--native" in sys.argv

    N = 1000
    alpha = 0.8
    dist = "uniform_100"
    seed = 42

    print(f"Generating {N} keys (alpha={alpha}, dist={dist})...")
    keys = gen_keys(N, seed=seed)
    values = gen_alpha_values(N, alpha, seed=seed, minority_dist=dist)
    n_classes = len(np.unique(values))
    print(f"  {n_classes} distinct classes")

    print("\nRunning learned LSF pipeline...")
    t0 = time.perf_counter()
    results = run_lsf_docker(keys, values, competitor="LSF", seed=seed, learned=True)
    elapsed = time.perf_counter() - t0

    if results is None:
        print(f"\nFAILED after {elapsed:.1f}s")
        sys.exit(1)

    converted = results_to_json(results, keys, values, "LSF")
    if not converted:
        print(f"\nNo Filtered-Huffman_Opt results found in {len(results)} raw results.")
        print("Raw results:")
        for r in results:
            print(f"  {r}")
        sys.exit(1)

    print(f"\nSUCCESS ({elapsed:.1f}s)")
    for entry in converted:
        mem = entry["memory"]
        bpk = mem.get("bits_per_key", 0) + mem.get("model_bits_per_key", 0)
        print(f"  {entry['method']}: {bpk:.2f} bits/key, "
              f"{entry['inference_ns']['mean']:.0f} ns/query")


if __name__ == "__main__":
    main()
