#!/bin/bash
set -euo pipefail

MODE="${1:-}"
DATA_DIR="${2:-/data/}"
DATASET="${3:-caramel}"
COMPETITOR="${4:-all}"

case "$MODE" in
    train-and-bench)
        echo "=== Training models on $DATASET ==="
        python3 /usr/local/bin/train_models.py "$DATA_DIR" "$DATASET"
        echo ""
        echo "=== Benchmarking $DATASET (competitor=$COMPETITOR) ==="
        ribbon_learned_bench -r "$DATA_DIR" -d "$DATASET" -c "$COMPETITOR"
        ;;
    bench-only)
        ribbon_learned_bench -r "$DATA_DIR" -d "$DATASET" -c "$COMPETITOR"
        ;;
    train-only)
        python3 /usr/local/bin/train_models.py "$DATA_DIR" "$DATASET"
        ;;
    *)
        echo "Usage: entrypoint.sh <mode> <data_dir> <dataset> [competitor]"
        echo "Modes: train-and-bench, bench-only, train-only"
        exit 1
        ;;
esac
