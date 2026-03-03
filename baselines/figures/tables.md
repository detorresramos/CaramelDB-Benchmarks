# Baseline Results — binary_fuse


### Memory (bits/key) — binary_fuse [uniform_100]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 |
| CSF+BinaryFuse (Optimal) | 5.22 | 4.36 | 3.43 | 2.51 | 1.43 | 0.80 | 0.24 |
| CSF+Bloom (Shibuya) | 5.52 | 4.79 | 3.74 | 2.73 | 1.52 | 0.84 | 0.24 |
| CSF+BinaryFuse (Optimal) v2 | 5.23 | 4.37 | 3.43 | 2.51 | 1.42 | 0.81 | 0.24 |
| CSF+Bloom (Shibuya) v2 | 5.52 | 4.79 | 3.74 | 2.73 | 1.52 | 0.84 | 0.24 |
| Java CSF (Sux4J) | 5.02 | 4.28 | 3.54 | 2.80 | 2.06 | 1.69 | 1.39 |
| Java MPH Table | 35.00 | 39.63 | 44.38 | 49.01 | 53.64 | 55.95 | 57.80 |
| Learned CSF | 4.39 | 3.72 | 3.35 | 2.17 | 1.20 | 0.67 | 0.25 |


### Memory (bits/key) — binary_fuse [unique]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 |
| CSF+BinaryFuse (Optimal) | 26.32 | 21.10 | 15.82 | 10.62 | 5.36 | 2.70 | 0.56 |
| CSF+Bloom (Shibuya) | 27.32 | 21.82 | 16.31 | 10.95 | 5.45 | 2.74 | 0.56 |
| CSF+BinaryFuse (Optimal) v2 | 26.32 | 21.10 | 15.82 | 10.62 | 5.36 | 2.70 | 0.56 |
| CSF+Bloom (Shibuya) v2 | 27.31 | 21.82 | 16.31 | 10.96 | 5.45 | 2.74 | 0.56 |
| Java CSF (Sux4J) | 41.92 | 33.64 | 25.40 | 17.23 | 9.14 | 5.14 | 2.01 |
| Java MPH Table | 43.89 | 46.67 | 49.45 | 52.22 | 55.00 | 56.40 | 57.88 |
| Learned CSF | 11.44 | 10.44 | 8.06 | 6.94 | 5.03 | 3.54 | 1.13 |


### Memory (bits/key) — binary_fuse [zipfian]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 | 95.11 |
| CSF+BinaryFuse (Optimal) | 4.58 | 3.89 | 3.12 | 2.33 | 1.37 | 0.79 | 0.22 |
| CSF+Bloom (Shibuya) | 4.88 | 4.13 | 3.41 | 2.42 | 1.42 | 0.83 | 0.22 |
| CSF+BinaryFuse (Optimal) v2 | 4.59 | 3.89 | 3.13 | 2.34 | 1.36 | 0.77 | 0.23 |
| CSF+Bloom (Shibuya) v2 | 4.89 | 4.13 | 3.42 | 2.43 | 1.40 | 0.81 | 0.23 |
| Java CSF (Sux4J) | 4.99 | 4.33 | 3.65 | 2.93 | 2.20 | 1.79 | 1.40 |
| Java MPH Table | 35.36 | 40.04 | 44.60 | 49.15 | 53.71 | 55.99 | 57.81 |
| Learned CSF | 4.86 | 4.00 | 3.60 | 2.39 | 1.70 | 1.40 | 0.38 |


### Avg Inference Time (ns) — binary_fuse [uniform_100]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 490 | 477 | 502 | 555 | 502 | 662 | 595 |
| CSF+BinaryFuse (Optimal) | 239 | 250 | 255 | 255 | 210 | 178 | 181 |
| CSF+Bloom (Shibuya) | 319 | 296 | 260 | 250 | 330 | 214 | 231 |
| CSF+BinaryFuse (Optimal) v2 | 174 | 97 | 108 | 95 | 83 | 79 | 77 |
| CSF+Bloom (Shibuya) v2 | 103 | 97 | 98 | 94 | 100 | 97 | 90 |
| Java CSF (Sux4J) | 346 | 570 | 542 | 337 | 295 | 326 | 319 |
| Java MPH Table | 1412 | 1280 | 1322 | 1505 | 1245 | 1551 | 1997 |
| Learned CSF | 6435 | 6180 | 6107 | 6160 | 5474 | 3199 | 3337 |


### Avg Inference Time (ns) — binary_fuse [unique]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 635 | 517 | 592 | 649 | 520 | 736 | 806 |
| CSF+BinaryFuse (Optimal) | 372 | 339 | 291 | 282 | 207 | 188 | 184 |
| CSF+Bloom (Shibuya) | 400 | 354 | 300 | 341 | 312 | 236 | 236 |
| CSF+BinaryFuse (Optimal) v2 | 179 | 212 | 194 | 156 | 175 | 115 | 91 |
| CSF+Bloom (Shibuya) v2 | 223 | 169 | 173 | 97 | 138 | 98 | 95 |
| Java CSF (Sux4J) | 667 | 1320 | 399 | 377 | 365 | 330 | 358 |
| Java MPH Table | 1363 | 1439 | 1400 | 1236 | 1383 | 1390 | 1505 |
| Learned CSF | 3997545 | 3116383 | 2395699 | 1431918 | 601561 | 243241 | 36641 |


### Avg Inference Time (ns) — binary_fuse [zipfian]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 533 | 482 | 991 | 630 | 340 | 561 | 477 |
| CSF+BinaryFuse (Optimal) | 265 | 285 | 299 | 239 | 234 | 180 | 188 |
| CSF+Bloom (Shibuya) | 283 | 266 | 289 | 247 | 243 | 229 | 222 |
| CSF+BinaryFuse (Optimal) v2 | 128 | 103 | 105 | 101 | 89 | 91 | 83 |
| CSF+Bloom (Shibuya) v2 | 136 | 101 | 112 | 100 | 121 | 97 | 105 |
| Java CSF (Sux4J) | 397 | 357 | 1065 | 355 | 336 | 333 | 321 |
| Java MPH Table | 1195 | 1371 | 1533 | 1344 | 1250 | 1323 | 1925 |
| Learned CSF | 76811 | 64949 | 54403 | 39387 | 23674 | 13530 | 5064 |


### Construction Time (s) — binary_fuse [uniform_100]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 0.014 | 0.012 | 0.012 | 0.013 | 0.012 | 0.013 | 0.012 |
| CSF+BinaryFuse (Optimal) | 0.043 | 0.042 | 0.037 | 0.027 | 0.020 | 0.018 | 0.014 |
| CSF+Bloom (Shibuya) | 0.042 | 0.036 | 0.034 | 0.027 | 0.022 | 0.019 | 0.017 |
| CSF+BinaryFuse (Optimal) v2 | 0.044 | 0.026 | 0.023 | 0.016 | 0.011 | 0.007 | 0.005 |
| CSF+Bloom (Shibuya) v2 | 0.028 | 0.025 | 0.022 | 0.016 | 0.012 | 0.009 | 0.007 |
| Java CSF (Sux4J) | 0.415 | 0.264 | 0.286 | 0.184 | 0.168 | 0.189 | 0.145 |
| Java MPH Table | 0.569 | 0.407 | 0.409 | 0.438 | 0.438 | 0.572 | 0.414 |
| Learned CSF | 5.173 | 5.298 | 5.559 | 5.480 | 5.783 | 5.733 | 7.321 |


### Construction Time (s) — binary_fuse [unique]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 0.012 | 0.014 | 0.012 | 0.013 | 0.013 | 0.013 | 0.013 |
| CSF+BinaryFuse (Optimal) | 0.081 | 0.080 | 0.076 | 0.047 | 0.035 | 0.025 | 0.018 |
| CSF+Bloom (Shibuya) | 0.084 | 0.070 | 0.078 | 0.051 | 0.038 | 0.026 | 0.020 |
| CSF+BinaryFuse (Optimal) v2 | 0.067 | 0.058 | 0.051 | 0.038 | 0.017 | 0.013 | 0.006 |
| CSF+Bloom (Shibuya) v2 | 0.072 | 0.055 | 0.042 | 0.032 | 0.016 | 0.013 | 0.007 |
| Java CSF (Sux4J) | 0.559 | 0.518 | 0.619 | 0.339 | 0.238 | 0.198 | 0.190 |
| Java MPH Table | 0.496 | 0.527 | 0.539 | 0.487 | 0.485 | 0.472 | 0.627 |
| Learned CSF | ~983 | 785.224 | 589.425 | 359.657 | 184.074 | 85.772 | 21.813 |


### Construction Time (s) — binary_fuse [zipfian]

| N=100k Method | a=0.5 | a=0.6 | a=0.7 | a=0.8 | a=0.9 | a=0.95 | a=0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ Hash Table | 0.014 | 0.015 | 0.013 | 0.012 | 0.012 | 0.013 | 0.012 |
| CSF+BinaryFuse (Optimal) | 0.036 | 0.042 | 0.035 | 0.023 | 0.017 | 0.017 | 0.013 |
| CSF+Bloom (Shibuya) | 0.035 | 0.052 | 0.035 | 0.027 | 0.018 | 0.019 | 0.016 |
| CSF+BinaryFuse (Optimal) v2 | 0.026 | 0.020 | 0.016 | 0.012 | 0.011 | 0.006 | 0.004 |
| CSF+Bloom (Shibuya) v2 | 0.026 | 0.024 | 0.017 | 0.013 | 0.010 | 0.008 | 0.005 |
| Java CSF (Sux4J) | 0.291 | 0.268 | 0.238 | 0.185 | 0.172 | 0.169 | 0.160 |
| Java MPH Table | 0.569 | 0.515 | 0.552 | 0.490 | 0.428 | 0.470 | 0.419 |
| Learned CSF | 32.064 | 30.174 | 26.529 | 22.474 | 17.973 | 13.284 | 8.633 |


---

# Genomics Datasets — binary_fuse


### Memory (bits/key) — Genomics Datasets

| Method | E. coli Sakai (N=5.3M, α=0.97) | SRR10211353 (N=9.8M, α=0.20) | C. elegans (N=69.7M, α=0.82) |
| --- | --- | --- | --- |
| C++ Hash Table | 152.0 | 152.0 | 152.0 |
| CSF+BinaryFuse (Optimal) | 0.30 | 4.20 | 1.23 |
| CSF+Bloom (Shibuya) | 0.34 | 4.23 | 1.36 |
| Java CSF (Sux4J) | 1.24 | 3.59 | 1.59 |
| Java MPH Table | 11.55 | 11.55 | 11.55 |
| Learned CSF | 0.33 | 3.49 | timeout |


### Avg Inference Time (ns) — Genomics Datasets

| Method | E. coli Sakai (N=5.3M, α=0.97) | SRR10211353 (N=9.8M, α=0.20) | C. elegans (N=69.7M, α=0.82) |
| --- | --- | --- | --- |
| C++ Hash Table | 812 | 904 | 994 |
| CSF+BinaryFuse (Optimal) | 335 | 457 | 394 |
| CSF+Bloom (Shibuya) | 432 | 594 | 385 |
| Java CSF (Sux4J) | 598 | 621 | 853 |
| Java MPH Table | 1,666 | 1,716 | 2,054 |
| Learned CSF | 1,725 | 8,312 | timeout |


### Construction Time (s) — Genomics Datasets

| Method | E. coli Sakai (N=5.3M, α=0.97) | SRR10211353 (N=9.8M, α=0.20) | C. elegans (N=69.7M, α=0.82) |
| --- | --- | --- | --- |
| C++ Hash Table | 1.4 | 2.6 | 18.5 |
| CSF+BinaryFuse (Optimal) | 0.7 | 4.3 | 13.7 |
| CSF+Bloom (Shibuya) | 0.9 | 4.4 | 15.1 |
| Java CSF (Sux4J) | 1.3 | 4.6 | 17.8 |
| Java MPH Table | 8.1 | 13.6 | 136.3 |
| Learned CSF | 127.4 | 501.4 | timeout |
