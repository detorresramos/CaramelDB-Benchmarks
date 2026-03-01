"""Convert CaramelDB-style key/value data to LSF's .lrbin format.

Usage:
    python convert_to_lrbin.py keys.txt values.bin output_dir/dataset_name

Writes:
    output_dir/dataset_name_X.lrbin  (features: hashed keys as float32)
    output_dir/dataset_name_y.lrbin  (labels: uint16)
"""

import os
import struct
import sys
import hashlib

import numpy as np


def keys_to_features(keys):
    """Hash string keys to float32 features in [0, 1).

    Uses two hash functions to produce 2 float features per key.
    The features have no learnable structure (by design — this is the point).
    """
    features = np.zeros((len(keys), 2), dtype=np.float32)
    for i, key in enumerate(keys):
        h = hashlib.md5(key.encode()).digest()
        features[i, 0] = struct.unpack("<I", h[0:4])[0] / (2**32)
        features[i, 1] = struct.unpack("<I", h[4:8])[0] / (2**32)
    return features


def write_lrbin(keys, values, output_prefix):
    """Write keys/values in LSF's .lrbin format.

    X file: [num_examples: uint64] [num_features: uint64] [float32 data...]
    y file: [num_classes: uint16] [uint16 labels...]
    """
    features = keys_to_features(keys)
    num_examples, num_features = features.shape

    unique_values = np.unique(values)
    value_to_label = {v: i for i, v in enumerate(sorted(unique_values))}
    labels = np.array([value_to_label[v] for v in values], dtype=np.uint16)
    num_classes = len(unique_values)

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    with open(f"{output_prefix}_X.lrbin", "wb") as f:
        f.write(struct.pack("<Q", num_examples))
        f.write(struct.pack("<Q", num_features))
        f.write(features.tobytes())

    with open(f"{output_prefix}_y.lrbin", "wb") as f:
        f.write(struct.pack("<H", num_classes))
        f.write(labels.tobytes())

    return num_examples, num_features, num_classes


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <keys.txt> <values.bin> <output_prefix>")
        sys.exit(1)

    keys_path, values_path, output_prefix = sys.argv[1:4]

    with open(keys_path) as f:
        keys = f.read().splitlines()

    values = np.frombuffer(
        open(values_path, "rb").read(), dtype=np.dtype(">u8")
    ).astype(np.uint32)

    n, nf, nc = write_lrbin(keys, values, output_prefix)
    print(f"Wrote {n} examples, {nf} features, {nc} classes to {output_prefix}")


if __name__ == "__main__":
    main()
