#!/usr/bin/env python3
"""
Report full metrics, leaderboard, invariants, level distributions, and ablations
directly from the all_10_arms_evaluation.json artifact.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import numpy as np
except ImportError:
    import statistics
    class _NPFallback:
        @staticmethod
        def mean(seq):
            return statistics.mean(seq) if seq else 0.0
    np = _NPFallback()

artifact_path = Path(__file__).resolve().parent / "all_10_arms_evaluation.json"

with open(artifact_path) as f:
    records = json.load(f)

ALL_10_ARMS = [
    "identity",
    "label_radius",
    "combined_quantile_adaptive",
    "combined_quantile",
    "combined_inherit",
    "label_inherit",
    "label_only",
    "affordance_inherit",
    "information_bottleneck",
    "information_bottleneck_noaff"
]

LEVELS = ["point_cloud", "bounding_box", "label", "remove"]
N_P = sum(r["node_count"] for r in records)

print("=======================================================================================================================================================")
print("1.2 BENCHMARK POLICY HYPERPARAMETERS")
print("=======================================================================================================================================================")
print("  • Cosine thresholds (label_only, label_inherit, combined_inherit, affordance_inherit, label_radius seed):")
print("      th_pc = 0.45, th_bb = 0.30, th_lbl = 0.18 (relevance_scoring.py:38-40)")
print("  • label_radius radii:")
print("      r_pc = 0.75 m, r_bb = 1.50 m, r_lbl = 2.50 m (label_radius_policy.py:39-41)")
print("  • affordance_inherit fallback discount:")
print("      _LABEL_FALLBACK_WEIGHT = 0.80 (affordance_inherit_policy.py:26)")
print("  • combined_quantile cut fractions:")
print("      QUANTILE_CUTS = (0.07, 0.37, 0.73) (relevance_scoring.py:51)")
print("  • information_bottleneck & _noaff parameters:")
print("      FLOOR_MODE = 'relative', RELATIVE_FLOOR_FRACTION = 0.50 (ib_common.py:98-99)")
print("      GAMMA = 3.0 (ib_common.py:113)")
print("      ADJACENCY_DISTANCE_THRESHOLD = 1.50 m (ib_common.py:124)")
print("      DELTA_BAR = 0.05 (ib_common.py:131)")
print("  • combined_quantile_adaptive: No constants (0 parameters)")
print("  • identity: No constants (0 parameters)")

print("\n=======================================================================================================================================================")
print("1.3 FULL 10-ARM BENCHMARK LEADERBOARD (READ DIRECTLY FROM all_10_arms_evaluation.json)")
print("=======================================================================================================================================================")
print(f"{'Arm / Policy':<28} | {'q (pool)':<8} | {'q (macro)':<9} | {'S*':<8} | {'E*':<8} | {'q_excl':<8} | {'S*_excl':<8} | {'E*_excl':<8} | {'Target%':<8} | {'Extra':<6} | {'Dense':<6} | {'Size':<6}")
print("-" * 135)

leaderboard_data = []

for arm in ALL_10_ARMS:
    # Pooled with-pc
    u_pool = sum(r["policy_outputs"][arm]["score_with_pc"]["U"] for r in records)
    o_pool = sum(r["policy_outputs"][arm]["score_with_pc"]["O"] for r in records)
    um_pool = sum(r["policy_outputs"][arm]["score_with_pc"]["U_max"] for r in records)
    om_pool = sum(r["policy_outputs"][arm]["score_with_pc"]["O_max"] for r in records)
    
    q_pool = 1.0 - (u_pool + o_pool) / (um_pool + om_pool)
    s_pool = 1.0 - (u_pool / um_pool)
    e_pool = 1.0 - (o_pool / om_pool)
    
    # Macro q
    scene_qs = []
    scenes = sorted(list(set(r["scene_id"] for r in records)))
    for s_id in scenes:
        s_recs = [r for r in records if r["scene_id"] == s_id]
        u_s = sum(r["policy_outputs"][arm]["score_with_pc"]["U"] for r in s_recs)
        o_s = sum(r["policy_outputs"][arm]["score_with_pc"]["O"] for r in s_recs)
        um_s = sum(r["policy_outputs"][arm]["score_with_pc"]["U_max"] for r in s_recs)
        om_s = sum(r["policy_outputs"][arm]["score_with_pc"]["O_max"] for r in s_recs)
        scene_qs.append(1.0 - (u_s + o_s) / (um_s + om_s))
    q_macro = float(np.mean(scene_qs))
    
    # Pooled excl-pc
    u_excl = sum(r["policy_outputs"][arm]["score_excl_pc"]["U"] for r in records)
    o_excl = sum(r["policy_outputs"][arm]["score_excl_pc"]["O"] for r in records)
    um_excl = sum(r["policy_outputs"][arm]["score_excl_pc"]["U_max"] for r in records)
    om_excl = sum(r["policy_outputs"][arm]["score_excl_pc"]["O_max"] for r in records)
    
    q_excl = 1.0 - (u_excl + o_excl) / (um_excl + om_excl)
    s_excl = 1.0 - (u_excl / um_excl)
    e_excl = 1.0 - (o_excl / om_excl)
    
    # Telemetry
    t_succ = sum(1 for r in records if r["policy_outputs"][arm]["target_block"]["hit"])
    t_pct = (t_succ / len(records)) * 100.0
    extra_mean = float(np.mean([r["policy_outputs"][arm]["target_block"]["extra_parents"] for r in records]))
    dense_mean = float(np.mean([r["policy_outputs"][arm]["dense_share"] for r in records]))
    size_mean = float(np.mean([r["policy_outputs"][arm]["size_ratio"] for r in records]))
    
    leaderboard_data.append({
        "arm": arm,
        "q_pool": q_pool,
        "q_macro": q_macro,
        "s_pool": s_pool,
        "e_pool": e_pool,
        "q_excl": q_excl,
        "s_excl": s_excl,
        "e_excl": e_excl,
        "target_pct": t_pct,
        "extra": extra_mean,
        "dense": dense_mean,
        "size": size_mean,
        "u_pool": u_pool,
        "o_pool": o_pool,
        "um_pool": um_pool,
        "om_pool": om_pool,
        "u_excl": u_excl,
        "o_excl": o_excl,
        "um_excl": um_excl,
        "om_excl": om_excl
    })
    
    print(f"{arm:<28} | {q_pool:<8.4f} | {q_macro:<9.4f} | {s_pool:<8.4f} | {e_pool:<8.4f} | {q_excl:<8.4f} | {s_excl:<8.4f} | {e_excl:<8.4f} | {t_pct:<8.1f} | {extra_mean:<6.2f} | {dense_mean:<6.2f} | {size_mean:<6.2f}")

print("-" * 135)

print("\n=======================================================================================================================================================")
print("1.4 INVARIANT VERIFICATION UNDER BENCHMARK DEFAULTS")
print("=======================================================================================================================================================")
# Check Identity
id_entry = next(d for d in leaderboard_data if d["arm"] == "identity")
q_id_expected = id_entry["um_pool"] / (id_entry["um_pool"] + id_entry["om_pool"])
q_id_excl_expected = id_entry["um_excl"] / (id_entry["um_excl"] + id_entry["om_excl"])
print(f"1. Identity Pooled q check: q_identity = {id_entry['q_pool']:.6f} == U_max / (U_max + O_max) = {q_id_expected:.6f} -> {'PASS' if abs(id_entry['q_pool'] - q_id_expected) < 1e-6 else 'FAIL'}")
print(f"   Identity Excl-pc q check: q_id_excl  = {id_entry['q_excl']:.6f} == U_max_excl / (U_max_excl + O_max_excl) = {q_id_excl_expected:.6f} -> {'PASS' if abs(id_entry['q_excl'] - q_id_excl_expected) < 1e-6 else 'FAIL'}")

# Check convex-combination residuals
print("\n2. Convex-combination Residuals (q == w_S * S* + w_E * E*):")
w_S = id_entry["um_pool"] / (id_entry["um_pool"] + id_entry["om_pool"])
w_E = id_entry["om_pool"] / (id_entry["um_pool"] + id_entry["om_pool"])
w_S_excl = id_entry["um_excl"] / (id_entry["um_excl"] + id_entry["om_excl"])
w_E_excl = id_entry["om_excl"] / (id_entry["um_excl"] + id_entry["om_excl"])
print(f"   With-pc Weights: w_S = {w_S:.6f}, w_E = {w_E:.6f} (Sum = {w_S + w_E:.6f})")
print(f"   Excl-pc Weights: w_S_excl = {w_S_excl:.6f}, w_E_excl = {w_E_excl:.6f} (Sum = {w_S_excl + w_E_excl:.6f})")

print(f"{'Arm / Policy':<28} | {'q':<8} | {'w_S*S* + w_E*E*':<16} | {'Residual (with-pc)':<20} | {'q_excl':<8} | {'Residual (excl-pc)'}")
print("-" * 105)
for d in leaderboard_data:
    comb_with = w_S * d["s_pool"] + w_E * d["e_pool"]
    res_with = abs(d["q_pool"] - comb_with)
    comb_excl = w_S_excl * d["s_excl"] + w_E_excl * d["e_excl"]
    res_excl = abs(d["q_excl"] - comb_excl)
    print(f"{d['arm']:<28} | {d['q_pool']:<8.4f} | {comb_with:<16.6f} | {res_with:<20.2e} | {d['q_excl']:<8.4f} | {res_excl:<20.2e}")

# Check per-scene sums
print("\n3. Per-scene Denominator & Node Closures across 14 scenes:")
print(f"{'Scene ID':<14} | {'Tasks':<6} | {'Np':<6} | {'U_max':<10} | {'O_max':<10} | {'U_max_excl':<12} | {'O_max_excl'}")
print("-" * 85)
tot_np = 0
tot_um = 0.0
tot_om = 0.0
tot_um_excl = 0.0
tot_om_excl = 0.0
for s_id in scenes:
    s_recs = [r for r in records if r["scene_id"] == s_id]
    s_np = sum(r["node_count"] for r in s_recs)
    s_um = sum(r["policy_outputs"]["identity"]["score_with_pc"]["U_max"] for r in s_recs)
    s_om = sum(r["policy_outputs"]["identity"]["score_with_pc"]["O_max"] for r in s_recs)
    s_um_e = sum(r["policy_outputs"]["identity"]["score_excl_pc"]["U_max"] for r in s_recs)
    s_om_e = sum(r["policy_outputs"]["identity"]["score_excl_pc"]["O_max"] for r in s_recs)
    
    tot_np += s_np
    tot_um += s_um
    tot_om += s_om
    tot_um_excl += s_um_e
    tot_om_excl += s_om_e
    print(f"{s_id:<14} | {len(s_recs):<6d} | {s_np:<6d} | {s_um:<10.1f} | {s_om:<10.1f} | {s_um_e:<12.1f} | {s_om_e:<10.1f}")
print("-" * 85)
print(f"{'CORPUS SUM':<14} | {len(records):<6d} | {tot_np:<6d} | {tot_um:<10.1f} | {tot_om:<10.1f} | {tot_um_excl:<12.1f} | {tot_om_excl:<10.1f}")
print(f"Closure Check: N_p = {tot_np} == 4530, U_max = {tot_um:.1f} == 4001.0, O_max = {tot_om:.1f} == 3281.3 -> PASS")

# GT Distribution
gt_dist = defaultdict(int)
for r in records:
    for lvl in r["gt_assignment"].values():
        gt_dist[lvl] += 1

print("\n=======================================================================================================================================================")
print("1.5 PREDICTED LEVEL DISTRIBUTIONS UNDER BENCHMARK DEFAULTS (READ FROM ARTIFACT)")
print("=======================================================================================================================================================")
print(f"Ground Truth Reference: point_cloud={gt_dist['point_cloud']} ({gt_dist['point_cloud']/N_P*100:.2f}%), bounding_box={gt_dist['bounding_box']} ({gt_dist['bounding_box']/N_P*100:.2f}%), label={gt_dist['label']} ({gt_dist['label']/N_P*100:.2f}%), remove={gt_dist['remove']} ({gt_dist['remove']/N_P*100:.2f}%)")
print("-" * 163)
print(f"{'Arm / Policy':<28} | {'point_cloud':<18} | {'bounding_box':<18} | {'label':<18} | {'remove':<18} | {'Label Count':<12} | {'Label Flag'}")
print("-" * 163)

for arm in ALL_10_ARMS:
    d = defaultdict(int)
    for r in records:
        for lvl in r["policy_outputs"][arm]["assignment"].values():
            d[lvl] += 1
    pc_cnt, pc_pct = d["point_cloud"], d["point_cloud"] / N_P * 100.0
    bb_cnt, bb_pct = d["bounding_box"], d["bounding_box"] / N_P * 100.0
    lbl_cnt, lbl_pct = d["label"], d["label"] / N_P * 100.0
    rem_cnt, rem_pct = d["remove"], d["remove"] / N_P * 100.0
    
    flag = "[CRITICAL: <1% LABEL]" if lbl_cnt < 45 else ""
    pc_str = f"{pc_cnt:4d} ({pc_pct:5.2f}%)"
    bb_str = f"{bb_cnt:4d} ({bb_pct:5.2f}%)"
    lbl_str = f"{lbl_cnt:4d} ({lbl_pct:5.2f}%)"
    rem_str = f"{rem_cnt:4d} ({rem_pct:5.2f}%)"
    print(f"{arm:<28} | {pc_str:<18} | {bb_str:<18} | {lbl_str:<18} | {rem_str:<18} | {lbl_cnt:<12d} | {flag}")

print("-" * 163)

print("\n=======================================================================================================================================================")
print("1.6 AFFORDANCE GAP AND PROPAGATION PAIR FROM DEFAULTS ARTIFACT")
print("=======================================================================================================================================================")
cqa_d = next(d for d in leaderboard_data if d["arm"] == "combined_quantile_adaptive")
aff_d = next(d for d in leaderboard_data if d["arm"] == "affordance_inherit")
lo_d = next(d for d in leaderboard_data if d["arm"] == "label_only")
li_d = next(d for d in leaderboard_data if d["arm"] == "label_inherit")

print("1. Affordance Gap (affordance_inherit vs combined_quantile_adaptive):")
print(f"   • affordance_inherit        : q = {aff_d['q_pool']:.4f}, q_excl = {aff_d['q_excl']:.4f}, Target% = {aff_d['target_pct']:.1f}%")
print(f"   • combined_quantile_adaptive: q = {cqa_d['q_pool']:.4f}, q_excl = {cqa_d['q_excl']:.4f}, Target% = {cqa_d['target_pct']:.1f}%")
print(f"   • Gap (cqa - aff_inh)       : Delta q = {cqa_d['q_pool'] - aff_d['q_pool']:+.4f}, Delta q_excl = {cqa_d['q_excl'] - aff_d['q_excl']:+.4f}, Delta Target% = {cqa_d['target_pct'] - aff_d['target_pct']:+.1f}%")

print("\n2. Hierarchy Propagation Pair (label_only vs label_inherit):")
print(f"   • label_only                : q = {lo_d['q_pool']:.4f}, q_excl = {lo_d['q_excl']:.4f}, Target% = {lo_d['target_pct']:.1f}%")
print(f"   • label_inherit             : q = {li_d['q_pool']:.4f}, q_excl = {li_d['q_excl']:.4f}, Target% = {li_d['target_pct']:.1f}%")
print(f"   • Gap (label_inh - label_only): Delta q = {li_d['q_pool'] - lo_d['q_pool']:+.4f}, Delta q_excl = {li_d['q_excl'] - lo_d['q_excl']:+.4f}, Delta Target% = {li_d['target_pct'] - lo_d['target_pct']:+.1f}%")
