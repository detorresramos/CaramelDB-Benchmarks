#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Theory Validation ==="
python "$DIR/theory_validation/run_experiments.py"
python "$DIR/theory_validation/make_plots.py"

echo "=== Baselines ==="
python "$DIR/baselines/run_baselines.py"
python "$DIR/baselines/make_plots.py"

echo "=== Genomics Baselines (skip if datasets not found) ==="
for csv in "$DIR"/data/*_k15.csv; do
    [ -f "$csv" ] && python "$DIR/baselines/run_baselines.py" --dataset "$csv"
done

echo "=== Shibuya Comparison ==="
python "$DIR/shibuya_comparison/run_experiments.py"
python "$DIR/shibuya_comparison/make_plots.py"

echo "=== Paper Figures ==="
python "$DIR/baselines/paper_plots.py"

echo "Done. Figures written to:"
echo "  $DIR/theory_validation/figures/"
echo "  $DIR/baselines/figures/"
echo "  $DIR/shibuya_comparison/figures/"
