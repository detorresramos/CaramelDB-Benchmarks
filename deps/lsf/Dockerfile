# Builds ribbon_learned_bench and includes Python+TF for model training.
# Build context is the deps/ directory.
#
# Usage:
#   docker build --platform linux/amd64 -t caramel-lsf -f deps/lsf/Dockerfile deps/
#
#   # Train + benchmark:
#   docker run --platform linux/amd64 -v /tmp/data:/data caramel-lsf \
#       train-and-bench /data/ caramel all
#
#   # Benchmark only (no learned models):
#   docker run --platform linux/amd64 -v /tmp/data:/data caramel-lsf \
#       bench-only /data/ caramel CSF

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
    llvm-17 clang-17 libc++-17-dev libc++abi-17-dev \
    libtbb-dev python3 && \
    rm -rf /var/lib/apt/lists/*

# Copy LSF source (just what CMake needs, not the huge submodules)
COPY LearnedStaticFunction/CMakeLists.txt \
     LearnedStaticFunction/ribbon_learned_bench.cpp \
     LearnedStaticFunction/filter_tuner.cpp \
     LearnedStaticFunction/plot_model_calibration.cpp \
     /lsf/
COPY LearnedStaticFunction/include/ /lsf/include/

# Clone submodules with shallow clones
WORKDIR /lsf/lib
RUN git clone --depth 1 https://github.com/tensorflow/tensorflow.git \
    --branch v2.19.0 --single-branch tensorflow
RUN git clone --depth 1 https://github.com/stefanfred/BuRR.git \
    --branch falsepositiveVLR --single-branch burr-vl
RUN git clone --depth 1 https://github.com/Maratyszcza/psimd.git psimd

WORKDIR /lsf/lib/burr-vl
RUN git submodule update --init --recursive --depth 1

# Build ribbon_learned_bench
WORKDIR /lsf/build
RUN cmake .. \
    -DCMAKE_C_COMPILER=clang-17 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_ENABLE_XNNPACK=OFF && \
    cmake --build . --target ribbon_learned_bench -j$(nproc)

# --- Runtime image ---
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libc++-17-dev libc++abi-17-dev libtbb12 \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    tensorflow==2.19.0 \
    keras==3.10.0 \
    numpy \
    scikit-learn

COPY --from=builder /lsf/build/ribbon_learned_bench /usr/local/bin/ribbon_learned_bench
COPY --from=builder /lsf/build/libRibbonSorter.so /usr/local/lib/libRibbonSorter.so
RUN ldconfig

COPY lsf/train_models.py /usr/local/bin/train_models.py
COPY lsf/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
