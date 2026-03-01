#!/usr/bin/env bash
# Install all dependencies for CaramelDB-Benchmarks on Ubuntu 24.04 x86_64.
#
# Usage:
#   sudo ./setup.sh          # install system packages + build everything
#   ./setup.sh --no-system   # skip apt-get, only build deps (if packages already installed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
INSTALL_SYSTEM=true
for arg in "$@"; do
    case "$arg" in
        --no-system) INSTALL_SYSTEM=false ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
if $INSTALL_SYSTEM; then
    echo "=== Installing system packages ==="
    apt-get update && apt-get install -y \
        build-essential cmake git python3 python3-pip python3-venv \
        openjdk-17-jdk maven \
        llvm-17 clang-17 libc++-17-dev libc++abi-17-dev libtbb-dev
fi

# ---------------------------------------------------------------------------
# Python venv
# ---------------------------------------------------------------------------
echo "=== Setting up Python venv ==="
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib tqdm scikit-learn tensorflow

# ---------------------------------------------------------------------------
# Build CaramelDB (expected as sibling directory)
# ---------------------------------------------------------------------------
CARAMEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/CaramelDB"
if [ -d "$CARAMEL_DIR" ]; then
    echo "=== Building CaramelDB ==="
    (cd "$CARAMEL_DIR" && python bin/build.py)
else
    echo "WARNING: CaramelDB not found at $CARAMEL_DIR"
    echo "  Clone it as a sibling directory and run: cd CaramelDB && python bin/build.py"
fi

# ---------------------------------------------------------------------------
# Java JARs
# ---------------------------------------------------------------------------
echo "=== Building Java dependencies ==="
(cd deps/java/java-caramel && mvn -q package -DskipTests)
(cd deps/java/java-mph && mvn -q package -DskipTests)

# ---------------------------------------------------------------------------
# LSF (native build — no Docker needed on x86)
# ---------------------------------------------------------------------------
echo "=== Building LSF (ribbon_learned_bench) ==="
LSF_DIR="deps/LearnedStaticFunction"
if [ -d "$LSF_DIR" ]; then
    mkdir -p "$LSF_DIR/build"
    (cd "$LSF_DIR/build" && \
        cmake .. \
            -DCMAKE_C_COMPILER=clang-17 \
            -DCMAKE_CXX_COMPILER=clang++-17 \
            -DCMAKE_BUILD_TYPE=Release \
            -DTFLITE_ENABLE_XNNPACK=OFF && \
        cmake --build . --target ribbon_learned_bench -j"$(nproc)")
else
    echo "WARNING: $LSF_DIR not found. Clone the LearnedStaticFunction submodule."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the venv with: source .venv/bin/activate"
