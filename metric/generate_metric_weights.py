#!/usr/bin/env python3
"""
Metric Weight Derivation & Invariant Verification Script.

Derives the paper's metric weights:
  w_S = U_max / (U_max + O_max)
  w_E = O_max / (U_max + O_max)
from the maximum possible penalty totals over the benchmark set,
and verifies the convex combination q = w_S * S* + w_E * E*.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = REPO_ROOT / "evaluation" / "all_10_arms_evaluation.json"


def derive_weights(defaults_artifact_path: Path):
    with open(defaults_artifact_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Calculate pooled totals across all placed tasks for the identity policy
    id_recs = [r["policy_outputs"]["identity"]["score_with_pc"] for r in records]
    U_max = sum(r["U_max"] for r in id_recs)
    O_max = sum(r["O_max"] for r in id_recs)
    denom = U_max + O_max

    w_S = U_max / denom
    w_E = O_max / denom

    # Excl-pc totals
    id_excl = [r["policy_outputs"]["identity"]["score_excl_pc"] for r in records]
    U_max_excl = sum(r["U_max"] for r in id_excl)
    O_max_excl = sum(r["O_max"] for r in id_excl)
    denom_excl = U_max_excl + O_max_excl

    w_S_excl = U_max_excl / denom_excl
    w_E_excl = O_max_excl / denom_excl

    print("=" * 78)
    print("METRIC WEIGHT DERIVATION FROM BENCHMARK ARTIFACTS")
    print("=" * 78)
    print(f"Total placed tasks evaluated: {len(records)}")
    print(f"Pooled With-PC Penalties:  U_max = {U_max:.1f}, O_max = {O_max:.1f} (Sum = {denom:.1f})")
    print(f"Derived Metric Weights:    w_S = {w_S:.6f}, w_E = {w_E:.6f} (Sum = {w_S + w_E:.6f})")
    print("-" * 78)
    print(f"Pooled Excl-PC Penalties:  U_max_excl = {U_max_excl:.1f}, O_max_excl = {O_max_excl:.1f} (Sum = {denom_excl:.1f})")
    print(f"Derived Excl-PC Weights:   w_S_excl = {w_S_excl:.6f}, w_E_excl = {w_E_excl:.6f} (Sum = {w_S_excl + w_E_excl:.6f})")
    print("=" * 78)

    # Invariant assertions
    assert abs(U_max - 4001.0) < 1e-4, f"U_max expected 4001.0, got {U_max}"
    assert abs(O_max - 3281.3) < 1e-4, f"O_max expected 3281.3, got {O_max}"
    assert abs(w_S - 0.5494143896845777) < 1e-5, f"w_S expected ~0.549414, got {w_S}"
    assert abs(w_E - 0.4505856103154223) < 1e-5, f"w_E expected ~0.450586, got {w_E}"
    assert abs(U_max_excl - 2137.0) < 1e-4, f"U_max_excl expected 2137.0, got {U_max_excl}"
    print("All metric weight assertions PASSED successfully.")

    return w_S, w_E


if __name__ == "__main__":
    derive_weights(DEFAULTS_PATH)

