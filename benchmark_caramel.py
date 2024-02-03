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


def empirical_entropy(x):
    unique_values, unique_counts = np.unique(x, return_counts=True)
    num_entries = np.sum(unique_counts)
    sorted_indices = unique_counts.argsort()
    sorted_values = unique_values[sorted_indices[::-1]]
    sorted_counts = unique_counts[sorted_indices[::-1]]
    sorted_probs = sorted_counts / num_entries
    return -1 * np.sum(sorted_probs * np.log2(sorted_probs)), sorted_probs[0], 


def get_data_metrics(keys, values):
    entropy, most_common_prob = empirical_entropy(values)
    if isinstance(keys[0], int):
        keys_size_bytes = len(keys) * 4 # 4 bytes per int
    elif isinstance(keys[0], str):
        keys_size_bytes = sum(len(key) for key in keys)
    else:
        raise ValueError("Invalid key type")
    values_size_bytes = values.shape[0] * values.shape[1] * 4 # 4 bytes per int
    return {
        "num_rows": values.shape[0],
        "num_columns": values.shape[1],
        "most_common_prob": most_common_prob,
        "keys_size_bytes": keys_size_bytes,
        "values_size_bytes": values_size_bytes,
        "total_size_bytes": keys_size_bytes + values_size_bytes,
        "entropy_bits": entropy, # TODO check this 
    }


def construct_caramel(keys, values):
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
    caramel_size_bytes = os.stat(savepath).st_size
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
