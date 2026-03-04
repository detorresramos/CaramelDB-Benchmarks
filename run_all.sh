#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Baselines ==="
python "$DIR/baselines/run_baselines.py"
python "$DIR/baselines/make_plots.py"

echo "Done. Figures written to:"
echo "  $DIR/baselines/figures/"
