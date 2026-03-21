# Learned CSF Benchmark Results: Genomics Datasets

## Dataset

- **Source:** E. coli (ecoli_sakai_k15)
- **Keys:** 5,326,642
- **Classes:** 42
- **Alpha:** 0.9696

## Configuration

- Three tokenizers x four architectures
- Quantization: float16 only

## Results

### md5 (2 features)

| Arch | Storage bpk | Model bpk | Total bpk | CE bpk | Query ns | Construct ms |
|------|-------------|-----------|-----------|--------|----------|--------------|
| L0_H0 | 0.3278 | 0.0004 | 0.3281 | 0.2440 | 1642.2 | 16459.4 |
| L1_H50 | 0.3266 | 0.0069 | 0.3335 | 0.2438 | 1784.5 | 28408.1 |
| L1_H100 | 0.3273 | 0.0136 | 0.3410 | 0.2436 | 1854.8 | 18565.7 |
| L2_H50 | 0.3278 | 0.0145 | 0.3423 | 0.2435 | 1914.1 | 19172.6 |

### kmer_ordinal (15 features)

| Arch | Storage bpk | Model bpk | Total bpk | CE bpk | Query ns | Construct ms |
|------|-------------|-----------|-----------|--------|----------|--------------|
| L1_H50 | 0.3127 | 0.0088 | 0.3216 | 0.2423 | 7843.4 | 76614.5 |
| L2_H50 | 0.3120 | 0.0165 | 0.3284 | 0.2420 | 1976.1 | 72205.9 |
| L1_H100 | 0.3117 | 0.0175 | 0.3292 | 0.2420 | 1927.8 | 19316.5 |
| L0_H0 | 0.3278 | 0.0020 | 0.3298 | 0.2443 | 6643.3 | 42814.3 |

### kmer_onehot (60 features)

| Arch | Storage bpk | Model bpk | Total bpk | CE bpk | Query ns | Construct ms |
|------|-------------|-----------|-----------|--------|----------|--------------|
| L1_H50 | 0.3072 | 0.0156 | 0.3228 | 0.2395 | 7966.9 | 58247.2 |
| L0_H0 | 0.3159 | 0.0077 | 0.3236 | 0.2425 | 8304.1 | 48442.3 |
| L2_H50 | 0.3094 | 0.0233 | 0.3326 | 0.2393 | 2197.6 | 19957.0 |
| L1_H100 | 0.3066 | 0.0311 | 0.3377 | 0.2388 | 2035.9 | 61095.9 |

## Environment

- **Training:** GPU (NVIDIA A10G, EC2)
- **Inference/Construction:** Mac (Apple Silicon)

## Summary

- **Best overall:** kmer_ordinal L1_H50 at 0.3216 total bpk
- **Best md5:** L0_H0 at 0.3281 total bpk
