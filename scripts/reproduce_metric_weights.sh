#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Deriving Metric Weights & Verifying Invariants ==="
python3 "$REPO_ROOT/metric/generate_metric_weights.py"

