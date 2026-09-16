#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Regenerating Paper Results Table from Evaluation Artifact ==="
python3 "$REPO_ROOT/evaluation/report_results.py"

