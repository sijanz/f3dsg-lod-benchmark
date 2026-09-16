#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Running Kinematic Constants Sensitivity Sweep ==="
python3 "$REPO_ROOT/evaluation/run_sensitivity_sweep.py"

