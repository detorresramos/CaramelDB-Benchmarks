"""Train models on GPU only — no C++ benchmark.

Usage:
    python deps/lsf/train_gpu.py <csv_path> [--tokenizer md5 kmer_ordinal kmer_onehot]

Converts CSV → .lrbin, trains all architectures, evaluates models,
saves results to model_cache. Transfer cache to Mac for C++ inference.
"""

import argparse
import csv
import os
import sys
import tempfile

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, ".."))

from lsf.convert_to_lrbin import write_lrbin
from lsf.run_benchmark import _cache_key, _cache_models, MODEL_CACHE_DIR, DATASET_NAME


def load_dataset(path):
    keys = []
    values = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if "key" in fieldnames and "value" in fieldnames:
            key_col, val_col = "key", "value"
        elif "kmer" in fieldnames and "count" in fieldnames:
            key_col, val_col = "kmer", "count"
        else:
            raise ValueError(f"CSV must have 'key'/'value' or 'kmer'/'count' columns, got: {fieldnames}")
        for row in reader:
            keys.append(row[key_col])
            values.append(int(row[val_col]))
    return keys, np.array(values, dtype=np.uint32)


def train_and_eval(keys, values, tokenizer, data_dir):
    """Convert to lrbin, train, evaluate, return cache_dir."""
    os.makedirs(data_dir, exist_ok=True)
    output_prefix = os.path.join(data_dir, DATASET_NAME)

    print(f"\n{'='*60}")
    print(f"Tokenizer: {tokenizer}")
    print(f"{'='*60}")

    print(f"Converting {len(keys)} keys to .lrbin (tokenizer={tokenizer})...")
    write_lrbin(keys, values, output_prefix, tokenizer=tokenizer)

    cache_dir = os.path.join(MODEL_CACHE_DIR, _cache_key(keys, values, tokenizer=tokenizer))

    # Train
    from lsf.train_models import main as train_main
    sys.argv = [sys.argv[0], data_dir, DATASET_NAME]
    train_main()

    # Evaluate
    from lsf.evaluate_models import main as eval_main
    sys.argv = [sys.argv[0], data_dir, DATASET_NAME]
    eval_main()

    # Cache
    _cache_models(data_dir, DATASET_NAME, cache_dir)
    print(f"\nCached to: {cache_dir}")
    return cache_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--tokenizer", nargs="+", default=["md5", "kmer_ordinal", "kmer_onehot"])
    args = parser.parse_args()

    print(f"Loading {args.csv_path}...")
    keys, values = load_dataset(args.csv_path)
    print(f"  {len(keys)} keys, {len(np.unique(values))} classes")

    for tok in args.tokenizer:
        with tempfile.TemporaryDirectory(prefix=f"lsf_train_{tok}_") as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            train_and_eval(keys, values, tok, data_dir)

    print("\nAll training complete. Transfer model_cache to Mac:")
    print(f"  scp -r deps/lsf/model_cache/ <mac>:CaramelDB-Benchmarks/deps/lsf/model_cache/")


if __name__ == "__main__":
    main()
