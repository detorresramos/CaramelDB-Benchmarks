#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$REPO_ROOT/deps"

echo "Building LSF Docker image (this may take a while on first run)..."
docker build --platform linux/amd64 -t caramel-lsf -f "$SCRIPT_DIR/Dockerfile" "$DEPS_DIR"
echo "Done. Image: caramel-lsf"
