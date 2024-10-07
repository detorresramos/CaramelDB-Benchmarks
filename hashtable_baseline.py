import os

configs = [
    (None, "synthetic_uniform_100k_rows.npy"),
    (None, "synthetic_uniform_10k_rows.npy"),
    (None, "synthetic_power_law_100k_rows.npy"),
    (None, "synthetic_power_law_10k_rows.npy"),

    (None, "genomes_minH_block100.npy"),
    (None, "proteomes_minH_block100.npy"),
    
    (None, "aol_terms_2.npy"), #TODO flatten
    (None, "malicious_phish_counts.npy"),

    (None, "amzn_tokenized_3m_uint32.npy"),
    (None, "msmarco_tokenized_3m_128_uint32.npy"),
    (None, "msmarco_tokenized_3m_512_uint16.npy"),
    (None, "msmarco_tokenized_8m_uint16.npy"),
    (None, "pile_tokenized_7m.npy"),

    ("abc_keys.txt", "abc_headlines_quantized_embeddings_M5_uint8.npy"),
    ("word2vec_keys.txt", "word2vec_quantized_embeddings_M5_uint8.npy"),
    ("sift_keys.txt", "sift_M4_quantized_uint8.npy"),
    (None, "yandex_1000000000.npy"),
]

import numpy as np

def read_values(values_path):
    if values_path.endswith(".npy"):
        return np.load(values_path)
    raise ValueError("Invalid values path")

def read_keys(keys_path, values_size):
    if keys_path is None:
        return [str(i) for i in range(values_size)]
    if keys_path.endswith(".txt"):
        with open(keys_path, "r") as f:
            lines = f.readlines()
            try:
                return [int(x.strip()) for x in lines]
            except:
                return [x.strip() for x in lines]
    raise ValueError("Invalid keys path")

def benchmark_hash_table(keys_path, values_path):
    values = read_values(values_path)
    keys = read_keys(keys_path, len(values))
    
    import time
    start = time.time()
    m = {k: v for k, v in zip(keys, values)}
    construction_time = time.time() - start

    total_time = 0
    for key in keys:
        start = time.perf_counter_ns()
        x = m[key]
        total_time += time.perf_counter_ns() - start
    
    query_time = total_time / len(keys)

    print(f"For dataset: {values_path} construction_time: {construction_time}, query_time: {query_time}")


if __name__ == "__main__":
    base_path = "/share/data/caramel/"
    for keys, values in configs:
        keys_path = base_path + keys if keys else None
        benchmark_hash_table(keys_path, base_path + values)