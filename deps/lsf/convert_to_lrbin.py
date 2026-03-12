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


CHAR_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def md5_features(keys):
    """MD5 hash → 2 random floats. No learnable structure."""
    features = np.zeros((len(keys), 2), dtype=np.float32)
    for i, key in enumerate(keys):
        h = hashlib.md5(key.encode()).digest()
        features[i, 0] = struct.unpack("<I", h[0:4])[0] / (2**32)
        features[i, 1] = struct.unpack("<I", h[4:8])[0] / (2**32)
    return features


# Backward compat alias
keys_to_features = md5_features


def kmer_ordinal_features(keys):
    """Ordinal encode each k-mer position: {A,C,G,T} → {0,1,2,3}/3. Returns (n, k) features."""
    k = len(keys[0])
    features = np.zeros((len(keys), k), dtype=np.float32)
    for i, key in enumerate(keys):
        for j, ch in enumerate(key):
            features[i, j] = CHAR_MAP.get(ch, 0) / 3.0
    return features


def kmer_onehot_features(keys):
    """One-hot encode each position: 4 bits per char. Returns (n, 4*k) features."""
    k = len(keys[0])
    features = np.zeros((len(keys), 4 * k), dtype=np.float32)
    for i, key in enumerate(keys):
        for j, ch in enumerate(key):
            features[i, 4 * j + CHAR_MAP.get(ch, 0)] = 1.0
    return features


TOKENIZERS = {
    "md5": md5_features,
    "kmer_ordinal": kmer_ordinal_features,
    "kmer_onehot": kmer_onehot_features,
}


def write_lrbin(keys, values, output_prefix, tokenizer="md5"):
    """Write keys/values in LSF's .lrbin format.

    X file: [num_examples: uint64] [num_features: uint64] [float32 data...]
    y file: [num_classes: uint16] [uint16 labels...]
    """
    features = TOKENIZERS[tokenizer](keys)
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
