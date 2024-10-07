import tqdm
import os
import pandas as pd
import numpy as np

df = pd.read_csv("/share/data/caramel/CHEN_AMAZON_DATA/hist.tsv", sep="\t")

def weighted_random_sample_multiple(df, num_samples, K):
    output_array = np.zeros((num_samples, K), dtype=int)
    for i in tqdm.tqdm(range(K)):
        sampled_indices = np.random.choice(df.index, size=num_samples, p=df['count'] / df['count'].sum())
        output_array[:, i] = sampled_indices
    return output_array

for N in [1_000_000, 1_900_000]:
    K = 100
    sampled_indices_array = weighted_random_sample_multiple(df, N, K)
    np.save(f"/share/data/caramel/CHEN_AMAZON_DATA/us_sampled_{N}.npy", sampled_indices_array)
    os.system(f"python3 benchmark_caramel.py --values /share/data/caramel/CHEN_AMAZON_DATA/us_sampled_{N}.npy")