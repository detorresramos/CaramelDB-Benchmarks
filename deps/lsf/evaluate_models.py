"""Evaluate trained TFLite models using batched inference.

Computes cross-entropy and model size for all models in a directory,
picks the best one, so the C++ binary only needs to run on that single model.

Usage:
    python deps/lsf/evaluate_models.py <data_dir> <dataset_name>

Prints the best model filename to stdout (for use with ribbon_learned_bench -m).
Also writes <data_dir>/<dataset_name>_model_eval.json with all model metrics.
"""

import json
import os
import struct
import sys

import numpy as np


def read_lrbin(data_dir, dataset_name):
    """Read .lrbin feature and label files."""
    x_path = os.path.join(data_dir, f"{dataset_name}_X.lrbin")
    y_path = os.path.join(data_dir, f"{dataset_name}_y.lrbin")

    with open(x_path, "rb") as f:
        num_examples = struct.unpack("<Q", f.read(8))[0]
        num_features = struct.unpack("<Q", f.read(8))[0]
        X = np.frombuffer(f.read(), dtype=np.float32).reshape(num_examples, num_features)

    with open(y_path, "rb") as f:
        num_classes = struct.unpack("<H", f.read(2))[0]
        y = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)

    return X, y, num_classes


def parse_eval_txt(eval_path):
    """Parse a _eval.txt file into a dict."""
    with open(eval_path) as f:
        line = f.read().strip()
    meta = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            try:
                v = float(v)
                if v == int(v):
                    v = int(v)
            except ValueError:
                pass
            meta[k] = v
    return meta


def evaluate_tflite(tflite_path, X, y, num_classes):
    """Run batched TFLite inference, return cross-entropy in bits per key."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    runner = interp.get_signature_runner("serving_default")

    batch_size = 100_000
    all_ce = 0.0
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_x = X[start:end]
        batch_y = y[start:end]

        probs = runner(input=batch_x)
        probs = list(probs.values())[0]

        if num_classes == 2:
            probs = np.column_stack([1 - probs, probs])

        # Cross-entropy: -log2(p(true_label))
        min_prob = 2.0 ** -31
        true_probs = probs[np.arange(len(batch_y)), batch_y]
        true_probs = np.maximum(true_probs, min_prob)
        all_ce += -np.sum(np.log2(true_probs))

    return all_ce / len(X)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <data_dir> <dataset_name>")
        sys.exit(1)

    data_dir = sys.argv[1]
    dataset_name = sys.argv[2]

    model_dir = os.path.join(data_dir, f"{dataset_name}_models")
    if not os.path.isdir(model_dir):
        print(f"No model directory: {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {dataset_name} from {data_dir}...", file=sys.stderr)
    X, y, num_classes = read_lrbin(data_dir, dataset_name)
    print(f"  {len(X)} examples, {X.shape[1]} features, {num_classes} classes", file=sys.stderr)

    # Find all .tflite files
    tflite_files = sorted(
        f for f in os.listdir(model_dir)
        if f.startswith(dataset_name) and f.endswith(".tflite")
    )

    if not tflite_files:
        print("No .tflite files found", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(tflite_files)} models...", file=sys.stderr)
    n = len(X)
    results = []

    for fname in tflite_files:
        tflite_path = os.path.join(model_dir, fname)
        eval_path = tflite_path + "_eval.txt"

        meta = parse_eval_txt(eval_path) if os.path.exists(eval_path) else {}

        model_bytes = os.path.getsize(tflite_path)
        model_bpk = 8.0 * model_bytes / n

        print(f"  {fname} ({model_bytes} bytes, {model_bpk:.3f} bpk)...", file=sys.stderr)
        ce_bpk = evaluate_tflite(tflite_path, X, y, num_classes)
        # cross-entropy is a lower bound on storage_bits; use it as estimate
        total_est = ce_bpk + model_bpk

        result = {
            "filename": fname,
            "model_bytes": model_bytes,
            "model_bpk": model_bpk,
            "cross_entropy_bpk": ce_bpk,
            "total_est_bpk": total_est,
            **meta,
        }
        results.append(result)
        print(f"    CE={ce_bpk:.4f} bpk, model={model_bpk:.4f} bpk, est_total={total_est:.4f} bpk", file=sys.stderr)

    # Pick best by estimated total
    best = min(results, key=lambda r: r["total_est_bpk"])
    print(f"\nBest model: {best['filename']} (est {best['total_est_bpk']:.4f} bpk)", file=sys.stderr)

    # Write full results (convert numpy types for JSON)
    def _jsonable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    for r in results:
        for k, v in r.items():
            r[k] = _jsonable(v)

    out_path = os.path.join(data_dir, f"{dataset_name}_model_eval.json")
    with open(out_path, "w") as f:
        json.dump({"best": best["filename"], "models": results}, f, indent=2)
    print(f"Wrote {out_path}", file=sys.stderr)

    # Print best model filename to stdout (for -m flag)
    print(best["filename"])


if __name__ == "__main__":
    main()
