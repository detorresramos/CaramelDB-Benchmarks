# CaramelDB Benchmarks

Experiments and baseline comparisons for [CaramelDB](https://github.com/detorresramos/CaramelDB).

## Quick Start

Clone, build CaramelDB, and install Python dependencies. This is all you need
to run the core experiments (paper plots, hash table baselines, shibuya
comparison).

```bash
git clone --recursive https://github.com/detorresramos/CaramelDB-Benchmarks.git
cd CaramelDB-Benchmarks

# Build CaramelDB and install Python bindings
deps/CaramelDB/bin/build.py

# Install Python experiment dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- CMake 3.14+
- C++17 compiler

## Running Experiments

### Paper Plots

Validates theoretical lower/upper bounds on bits/key saved against empirical
measurements across filter types (XOR, BinaryFuse, Bloom) and value
distributions (unique, Zipfian, uniform-100).

```bash
python paper_plots/run_experiments.py   # generate data
python paper_plots/make_plots.py        # generate figures
```

### Baselines

Compares CSF+filter memory and query performance against hash tables (Python
and C++). With optional dependencies installed (see below), also compares
against Java and learned CSF implementations. Missing baselines are
automatically skipped.

```bash
python baselines/run_baselines.py       # generate data
python baselines/make_plots.py          # generate figures and tables
```

### Shibuya Comparison

Head-to-head of theory-guided Bloom filter parameter selection vs Shibuya et
al.'s empirical entropy-based approach.

```bash
python shibuya_comparison/make_plots.py
```

### Run Everything

```bash
bash run_all.sh
```

## Optional Baselines

### Java (Sux4J CSF, MPH Table)

Requires JDK 21+ and Maven.

```bash
cd deps/java/java-caramel && mvn package -q && cd ../../..
cd deps/java/java-mph && mvn package -q && cd ../../..
```

Once built, `run_baselines.py` will automatically include the Java methods.

### Learned Static Function

The [LearnedStaticFunction](https://github.com/gvinciguerra/LearnedStaticFunction)
baseline requires TensorFlow, Bazel, Clang 17, and C++23. Due to the heavy
dependency chain, we recommend building via Docker:

```bash
# TODO: wrapper script for LSF benchmarks
```

## Structure

```
deps/
  CaramelDB/                # submodule — core library
  LearnedStaticFunction/    # submodule — learned CSF baseline
  java/                     # Java baseline implementations
shared/                     # shared utilities (data_gen, theory, measure, shibuya)
paper_plots/                # theory validation experiments
baselines/                  # baseline comparisons
shibuya_comparison/         # epsilon selection comparison
```
