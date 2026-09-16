#!/usr/bin/env python3
"""
Generates all_10_arms_evaluation.json across all 10 policies.
Evaluates the 155 placed benchmark tasks using canonical policy hyperparameter configurations.
"""

import json
import hashlib
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from policies.policies_registry import load_policy_function, ALL_POLICIES
from policies.dataset_adapters.fungraph_loader import FunGraphDatasetLoader
from policies.utils.graph_modifier import create_modified_graph

LEVELS = ["remove", "label", "bounding_box", "point_cloud"]
LEVEL_RANK = {l: i for i, l in enumerate(LEVELS)}

COST_MATRIX = {
    "remove":       {"remove": 0.0, "label": 0.1, "bounding_box": 0.3, "point_cloud": 1.0},
    "label":        {"remove": 1.0, "label": 0.0, "bounding_box": 0.2, "point_cloud": 0.6},
    "bounding_box": {"remove": 2.0, "label": 1.5, "bounding_box": 0.0, "point_cloud": 0.3},
    "point_cloud":  {"remove": 4.0, "label": 3.0, "bounding_box": 2.0, "point_cloud": 0.0},
}

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


def score_predictions(gt_assignment: dict, pred_assignment: dict, excl_pc: bool = False):
    U = 0.0
    O = 0.0
    U_max = 0.0
    O_max = 0.0
    confusion = {g: {p: 0 for p in LEVELS} for g in LEVELS}
    
    for nid, g_lvl in gt_assignment.items():
        if excl_pc and g_lvl == "point_cloud":
            continue
        p_lvl = pred_assignment.get(nid, "remove")
        confusion[g_lvl][p_lvl] += 1
        
        g_idx = LEVEL_RANK[g_lvl]
        p_idx = LEVEL_RANK[p_lvl]
        cost = COST_MATRIX[g_lvl][p_lvl]
        
        if g_idx > p_idx:
            U += cost
        elif g_idx < p_idx:
            O += cost
            
        U_max += COST_MATRIX[g_lvl]["remove"]
        O_max += COST_MATRIX[g_lvl]["point_cloud"]
        
    S_star = 1.0 - (U / U_max) if U_max > 0 else 1.0
    E_star = 1.0 - (O / O_max) if O_max > 0 else 1.0
    denom = U_max + O_max
    Q = 1.0 - ((U + O) / denom) if denom > 0 else 1.0
    
    return {
        "U": U, "O": O, "U_max": U_max, "O_max": O_max,
        "S_star": S_star, "E_star": E_star, "Q": Q,
        "confusion": confusion,
        "n_evaluated": sum(sum(r.values()) for r in confusion.values())
    }


def compute_target_block(gt_assignment: dict, pred_assignment: dict, edges: list):
    sources = {e["source_id"] for e in edges}
    gt_parents = {nid for nid, lvl in gt_assignment.items() if lvl == "point_cloud" and nid not in sources}
    parent_id = next(iter(gt_parents)) if gt_parents else None
        
    critical = {nid for nid, lvl in gt_assignment.items() if lvl == "point_cloud"}
    pred_pc = {nid for nid, lvl in pred_assignment.items() if lvl == "point_cloud"}
    pred_parents = {nid for nid in pred_pc if nid not in sources}
    
    hit = critical.issubset(pred_pc) if critical else True
    hit_parent = (parent_id in pred_pc) if parent_id else False
    extra_parents = len(pred_parents - {parent_id}) if parent_id else len(pred_parents)
    
    return {
        "hit": hit,
        "hit_parent": hit_parent,
        "extra_parents": extra_parents
    }


def load_corpus_data():
    dataset_dir = REPO_ROOT / "dataset" / "FunGraph3D"
    loader = FunGraphDatasetLoader(str(dataset_dir))
    
    # Load 232 tasks reference to retrieve placed tasks and ground truth detail assignments
    eval_232_path = REPO_ROOT / "data" / "all_232_tasks_evaluation.json"
    with open(eval_232_path, "r", encoding="utf-8") as f:
        all_eval = json.load(f)
        
    placed_tasks = [t for t in all_eval if not t.get("dropped", False)]
    
    scene_ids = sorted(list(set(t["scene_id"] for t in placed_tasks)))
    scene_graphs = {}
    for s_id in scene_ids:
        scene_graphs[s_id] = loader.load_scene(s_id)
        
    return scene_graphs, placed_tasks


def generate_evaluation_artifact(output_path: Path):
    scene_graphs, placed_tasks = load_corpus_data()
    print(f"Loaded {len(scene_graphs)} scene graphs and {len(placed_tasks)} placed tasks.")
    
    base_sizes = {}
    for s_id, sg in scene_graphs.items():
        id_fn = load_policy_function("identity")
        id_res = id_fn(sg, "test")
        mod_g, _ = create_modified_graph(sg, id_res)
        base_sizes[s_id] = len(json.dumps(mod_g.to_dict()))
        
    records = []
    for idx, t in enumerate(placed_tasks, 1):
        s_id = t["scene_id"]
        task_str = t["task"]
        sg = scene_graphs[s_id]
        gt_assign = t["gt_assignment"]
        edges = t.get("edges", [])
        
        record = {
            "scene_id": s_id,
            "task": task_str,
            "filename": t.get("filename", ""),
            "target_obj_id": t.get("target_obj_id", ""),
            "target_aff_id": t.get("target_aff_id", ""),
            "node_count": t.get("node_count", len(gt_assign)),
            "dropped": False,
            "drop_category": None,
            "drop_reason": None,
            "raw_drop_detail": None,
            "gt_assignment": gt_assign,
            "policy_outputs": {}
        }
        
        for arm in ALL_10_ARMS:
            policy_fn = load_policy_function(arm)
            p_res = policy_fn(sg, task_str)
            p_assign = {nid: lvl.value for nid, lvl in p_res.detail_assignment.items()}
            
            s_with = score_predictions(gt_assign, p_assign, excl_pc=False)
            s_excl = score_predictions(gt_assign, p_assign, excl_pc=True)
            t_block = compute_target_block(gt_assign, p_assign, edges)
            
            n_pc = sum(1 for lvl in p_assign.values() if lvl == "point_cloud")
            dense_share = n_pc / len(p_assign) if p_assign else 0.0
            
            mod_g, _ = create_modified_graph(sg, p_res)
            g_bytes = len(json.dumps(mod_g.to_dict()))
            size_ratio = g_bytes / base_sizes[s_id] if base_sizes[s_id] > 0 else 1.0
            
            n_clusters = len(set(p_res.merge_assignment.values())) if getattr(p_res, "merge_assignment", None) else len(p_assign)
            n_label = sum(1 for lvl in p_assign.values() if lvl == "label")
            label_share = n_label / len(p_assign) if p_assign else 0.0
            
            record["policy_outputs"][arm] = {
                "score_with_pc": s_with,
                "score_excl_pc": s_excl,
                "target_block": t_block,
                "dense_share": round(dense_share, 4),
                "size_ratio": round(size_ratio, 4),
                "n_clusters": n_clusters,
                "label_share": round(label_share, 4),
                "assignment": p_assign
            }
            
        records.append(record)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)
        
    with open(output_path, "rb") as fp:
        sha256 = hashlib.sha256(fp.read()).hexdigest()
    print(f"Generated {output_path} (SHA-256: {sha256})")
    return sha256


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "all_10_arms_evaluation.json"
    generate_evaluation_artifact(out_file)
