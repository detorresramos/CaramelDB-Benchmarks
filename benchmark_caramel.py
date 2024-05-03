import os
import time
import carameldb
from carameldb import Caramel
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--keys", type=str)
    parser.add_argument("--values", type=str)

    return parser.parse_args()


def read_values(values_path):
    if values_path.endswith(".npy"):
        return np.load(values_path)
    raise ValueError("Invalid values path")


def read_keys(keys_path, values_size):
    if keys_path is None:
        return [i for i in range(values_size)]
    if keys_path.endswith(".txt"):
        with open(keys_path, "r") as f:
            lines = f.readlines()
            try:
                return [int(x.strip()) for x in lines]
            except:
                return [x.strip() for x in lines]
    raise ValueError("Invalid keys path")


def single_empirical_entropy(x):
    unique_values, unique_counts = np.unique(x, return_counts=True)
    num_entries = np.sum(unique_counts)
    sorted_indices = unique_counts.argsort()
    sorted_counts = unique_counts[sorted_indices[::-1]]
    sorted_probs = sorted_counts / num_entries
    return -1 * np.sum(sorted_probs * np.log2(sorted_probs)), sorted_probs[0], 


def empirical_entropy(x):
    if len(x.shape) == 1:
        return single_empirical_entropy(x)
    assert len(x.shape) == 2
    entropy = 0
    for i in range(x.shape[1]):
        entropy += single_empirical_entropy(x[: ,i])[0]

    unique_values, unique_counts = np.unique(x, return_counts=True)
    num_entries = np.sum(unique_counts)
    sorted_indices = unique_counts.argsort()
    sorted_counts = unique_counts[sorted_indices[::-1]]
    return entropy, sorted_counts[0] / num_entries


def get_data_metrics(keys, values):
    entropy, most_common_prob = empirical_entropy(values)
    if isinstance(keys[0], int):
        keys_size_bytes = len(keys) * 4 # 4 bytes per int
    elif isinstance(keys[0], str):
        keys_size_bytes = sum(len(key) for key in keys)
    else:
        raise ValueError("Invalid key type")
    if len(values.shape) == 2:
        values_size_bytes = values.shape[0] * values.shape[1] * 4 # 4 bytes per int
    else:
        values_size_bytes = values.shape[0] * 4
    return {
        "entropy_bits": entropy,
        "total_size_mb": (keys_size_bytes + values_size_bytes) / 1e6,
        "total_size_gb": (keys_size_bytes + values_size_bytes) / 1e9,
        "num_rows": values.shape[0],
        "num_columns": values.shape[1] if len(values.shape) == 2 else 1,
        "most_common_prob": most_common_prob,
        "values_size_mb": values_size_bytes / 1e6,
        "keys_size_mb": keys_size_bytes/ 1e6,
    }


def construct_caramel(keys, values):
    if not isinstance(keys[0], str):
        keys = [key.to_bytes(4, "little") for key in keys]
    start = time.time()
    caramel = Caramel(keys, values, verbose=False)
    construction_time = time.time() - start

    #TODO should we measure this query time in C++
    total_query_time = 0
    num_queries = 1000
    for i in range(num_queries):
        key = keys[i % len(keys)]
        start = time.time()
        caramel.query(key)
        total_query_time += time.time() - start
    average_query_time = total_query_time / num_queries

    savepath = "save.caramel"
    caramel.save(savepath)
    if os.path.isfile(savepath):
        caramel_size_bytes = os.path.getsize(savepath)
    else:
        caramel_size_bytes = 0
        for csf in os.scandir(savepath):
            caramel_size_bytes += os.path.getsize(csf)
    os.system(f"rm -rf {savepath}")

    return {
        "caramel_size_bytes": caramel_size_bytes,
        "caramel_size_MB": caramel_size_bytes / 1e6,
        "caramel_size_GB": caramel_size_bytes / 1e9,
        "construction_time": construction_time,
        "average_query_time": average_query_time,
    }


if __name__ == "__main__":
    args = parse_args()

    values = read_values(args.values)

    keys = read_keys(args.keys, values_size=len(values))

    assert len(values) == len(keys)

    data_metrics = get_data_metrics(keys, values)
    print("Data Metrics: ", data_metrics)

    caramel_metrics = construct_caramel(keys, values)
    print("Caramel Metrics: ", caramel_metrics)

    with open("benchmark_numbers.txt", "a") as f:
        f.write(f"{args.keys} | {args.values}\n")
        f.write(f"{data_metrics}\n")
        f.write(f"{caramel_metrics}\n")
        f.write("\n")
