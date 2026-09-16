#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Aggregating Planning Benchmark Results across 3,720 Trials ==="
python3 "$REPO_ROOT/planning/aggregate_planning_results.py"

