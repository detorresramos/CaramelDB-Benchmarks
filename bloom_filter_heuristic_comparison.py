import os
import carameldb

import numpy as np
from scipy import stats

def single_empirical_entropy(x):
    unique_values, unique_counts = np.unique(x, return_counts=True)
    num_entries = np.sum(unique_counts)
    sorted_indices = unique_counts.argsort()
    sorted_counts = unique_counts[sorted_indices[::-1]]
    sorted_probs = sorted_counts / num_entries
    return -1 * np.sum(sorted_probs * np.log2(sorted_probs)), sorted_probs[0], 

def generate_distribution(desired_entropy, n_values):
    x = np.zeros(n_values, dtype=np.uint32)
    entropy = single_empirical_entropy(x)[0]
    index = 0
    while entropy < desired_entropy:
        for i in range(4000):
            x[index] = index
            index += 1
        entropy = single_empirical_entropy(x)[0]
    # print("Entropy", entropy)
    return x, entropy


def generate_distribution_zipfian(desired_entropy, rows):
    x = np.random.zipf(2, size=rows)
    entropy = single_empirical_entropy(x)[0]
    index = len(x) - 1
    while entropy < desired_entropy:
        for i in range(4000):
            x[index] = index
            index -= 1
        entropy = single_empirical_entropy(x)[0]
    return x, entropy

def get_custom_threshold(entropy, alpha):
    coeff = ((1 - alpha) / alpha) / np.log(2)
    Cbf = 1.44
    if entropy < 2:
        Ccsf = (0.22 * entropy * entropy) + (0.18 * entropy) + 1.16
    else:
        Ccsf = (1.1 * entropy) + 0.2
    return (Cbf / Ccsf) * coeff

import warnings
warnings.filterwarnings("ignore")

for entropy in range(1, 20, 2):
    N = 1_000_000
    keys = [str(i) for i in range(N)]
    values, calculated_entropy = generate_distribution_zipfian(entropy, N)
    _, count = stats.mode(values)
    alpha = count / len(values)
    custom_e0_threshold = get_custom_threshold(calculated_entropy, alpha)
    #
    csf_with_their_threshold = carameldb.Caramel(keys, values, use_bloom_filter=True, custom_threshold=custom_e0_threshold, verbose=False)
    csf_with_their_threshold.save("test.csf")
    csf_with_their_threshold_size = os.path.getsize("test.csf")
    csf_no_filter = carameldb.Caramel(keys, values, use_bloom_filter=False, verbose=False)
    csf_no_filter.save("test.csf")
    csf_no_filter_size = os.path.getsize("test.csf")
    csf_with_our_threshold = carameldb.Caramel(keys, values, use_bloom_filter=True, verbose=False)
    csf_with_our_threshold.save("test.csf")
    csf_with_our_threshold_size = os.path.getsize("test.csf")
    print(f"CSF for entropy={entropy} has alpha={alpha} their threshold is {custom_e0_threshold}")
    print(f"Size with their threshold={csf_with_their_threshold_size}, size with our threshold={csf_with_our_threshold_size}, and size with no bf={csf_no_filter_size}")
    print("--------------------------------------------------------")


