import os

configs = [
    # (None, "synthetic_uniform_100k_rows.npy"),
    # (None, "synthetic_uniform_10k_rows.npy"),
    # (None, "synthetic_power_law_100k_rows.npy"),
    # (None, "synthetic_power_law_10k_rows.npy"),

    (None, "genomes_minH_block100.npy"),
    (None, "proteomes_minH_block100.npy"),
    
    (None, "aol_terms_2.npy"), #TODO flatten
    (None, "malicious_phish_counts.npy"),

    # (None, "amzn_tokenized_3m_uint32.npy"),
    # (None, "msmarco_tokenized_3m_128_uint32.npy"),
    # (None, "msmarco_tokenized_3m_512_uint16.npy"),
    # (None, "msmarco_tokenized_8m_uint16.npy"),
    # (None, "pile_tokenized_7m.npy"),

    # ("abc_keys.txt", "abc_headlines_quantized_embeddings_M5_uint8.npy"),
    ("word2vec_keys.txt", "word2vec_quantized_embeddings_M5_uint8.npy"),
    # ("sift_keys.txt", "sift_M4_quantized_uint8.npy"),
    ("quantized_sentence_bert_keys.txt", "sentence_bert_embeddings_train.npy"),
    (None, "yandex_10000000.npy"),
    (None, "yandex_100000000.npy"),
    # (None, "yandex_1000000000.npy"),
]

if __name__ == "__main__":
    base_path = "/share/data/caramel/"
    for keys, values in configs:
        keys_str = f"--keys {base_path}{keys}" if keys != None else ""
        os.system(f"python3 benchmark_caramel.py {keys_str} --values {base_path}{values}")