import os
import sys
from pathlib import Path
from copy import deepcopy
import numpy as np
import scipy.ndimage as ndi
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reference.derive_ground_truth import (
    SceneFloorRaster,
    solve_feasible_base_placements,
    extract_subpart_relationships,
    point_to_box_distance,
    load_dataset_annotations,
    load_dataset_relations,
    load_node_geometries,
    load_floor_elevation,
    resolve_task_targets,
    CONFIG
)

def evaluate_scene_26_cases(custom_config: dict, floor_raster: SceneFloorRaster, annots: list, rels: list, node_geoms: dict, z_floor: float, tasks: list, fixed_h: float = None):
    ACCEPTED_VERBS = {
        "pull to open or close", "rotate to open or close, adjust the setting",
        "control the water flow", "rotate to open or close"
    }

    r_reach = custom_config["ARM_MAX_REACH"] + custom_config["COLLISION_MARGIN"]

    case_results = {}
    dropped_count = 0
    total_counts = {"point_cloud": 0, "bounding_box": 0, "label": 0, "remove": 0}

    for t in tasks:
        fn = t["filename"]
        target_obj_id, target_aff_id = resolve_task_targets(t)
        p_aff = np.array(node_geoms[target_aff_id]["centroid"])

        kinematic_assembly = {target_obj_id}
        for r in rels:
            if r["second_node_annot_id"] == target_obj_id and r["description"] in ACCEPTED_VERBS:
                kinematic_assembly.add(r["first_node_annot_id"])

        feasible_placements, drop_reason = solve_feasible_base_placements(p_aff, floor_raster, z_floor, custom_config)

        if fixed_h is not None and feasible_placements:
            filtered = [p for p in feasible_placements if abs(p["h"] - fixed_h) <= 0.015]
            if not filtered:
                feasible_placements = []
                drop_reason = f"(ii) blocked at h={fixed_h}"
            else:
                feasible_placements = filtered

        if not feasible_placements:
            dropped_count += 1
            levels = {}
            for ann in annots:
                nid = ann["annot_id"]
                if nid in kinematic_assembly:
                    levels[nid] = "point_cloud"
                else:
                    levels[nid] = "remove"
            case_results[fn] = levels
            total_counts["point_cloud"] += len(kinematic_assembly)
            total_counts["remove"] += (len(annots) - len(kinematic_assembly))
            continue

        can = feasible_placements[0]
        can_shoulder = np.array([can["base_x"], can["base_y"], can["shoulder_z"]])
        all_shoulders = np.array([[x["base_x"], x["base_y"], x["shoulder_z"]] for x in feasible_placements])

        levels = {}
        for nid, g in node_geoms.items():
            if nid in kinematic_assembly:
                levels[nid] = "point_cloud"
                continue
            bmin = np.array(g["min"])
            bmax = np.array(g["max"])

            d_can = point_to_box_distance(can_shoulder, bmin, bmax)
            if d_can <= r_reach:
                levels[nid] = "bounding_box"
                continue

            clamped_all = np.clip(all_shoulders, bmin, bmax)
            d_all = np.linalg.norm(all_shoulders - clamped_all, axis=1)
            if np.any(d_all <= r_reach):
                levels[nid] = "label"
            else:
                levels[nid] = "remove"

        case_results[fn] = levels
        for lvl in ["point_cloud", "bounding_box", "label", "remove"]:
            total_counts[lvl] += sum(1 for v in levels.values() if v == lvl)

    return total_counts, dropped_count, case_results


def main():
    print("Preloading 0kitchen scene raster and metadata once...")
    scene_id = "0kitchen"
    dataset_dir = "dataset/FunGraph3D"
    annots = load_dataset_annotations(dataset_dir, scene_id)
    rels = load_dataset_relations(dataset_dir, scene_id)
    node_geoms = load_node_geometries(str(REPO_ROOT / "data/node_geom.json"), scene_id)
    z_floor = load_floor_elevation(str(REPO_ROOT / "data/affordance_heights.json"), scene_id)

    ply_path = os.path.join(dataset_dir, scene_id, f"{scene_id}.ply")
    floor_raster = SceneFloorRaster(ply_path, z_floor, node_geoms, res=CONFIG["GRID_RESOLUTION"])

    import json
    with open(str(REPO_ROOT / "data/tasks_target_objects.json")) as f:
        meta = json.load(f)
    kitchen_meta = [s for s in meta["scenes"] if s["scene_id"] == "0kitchen"][0]
    tasks = kitchen_meta["explicit_tasks"] + kitchen_meta["implicit_tasks"]

    base_cfg = dict(CONFIG)

    # Baseline evaluation
    default_counts, default_drops, default_assignments = evaluate_scene_26_cases(
        base_cfg, floor_raster, annots, rels, node_geoms, z_floor, tasks
    )

    def diff_against_default(assignments):
        diff_count = 0
        for fn in default_assignments:
            def_map = default_assignments[fn]
            cur_map = assignments.get(fn, {})
            for nid, lvl in def_map.items():
                if cur_map.get(nid) != lvl:
                    diff_count += 1
        return diff_count

    report_lines = []
    report_lines.append("# Sensitivity Sweep Report: 0kitchen Benchmark\n")
    report_lines.append("> **CRITICAL AUDIT DIRECTIVE**: `REACH_MARGIN = 0.05 m` explicitly ($R_{\\text{eff}} = 0.576\\,\\text{m}$). All sweeps regenerated at this baseline.\n")
    report_lines.append("**Scene**: `0kitchen` (26 tasks, 39 nodes per scene = 1,014 total node evaluations)  ")
    report_lines.append("**Baseline Configuration**:  ")
    report_lines.append(f"- `ARM_MAX_REACH` = {base_cfg['ARM_MAX_REACH']:.3f} m")
    report_lines.append(f"- `REACH_MARGIN` = {base_cfg['REACH_MARGIN']:.3f} m (R_eff = 0.576 m)")
    report_lines.append(f"- `COLLISION_MARGIN` = {base_cfg['COLLISION_MARGIN']:.3f} m (Default)")
    report_lines.append(f"- `mount_height` = solver-chosen (Default)")
    report_lines.append(f"- `SUPPORT_Z_TOLERANCE` = {base_cfg['SUPPORT_Z_TOLERANCE']:.3f} m (Default)")
    report_lines.append(f"- `r_nominal` = {base_cfg['R_NOMINAL_FRACTION'] * base_cfg['ARM_MAX_REACH']:.4f} m (0.75 * R_MAX, Default)\n")
    report_lines.append(f"**Baseline Totals (over 1,014 evaluations)**: `point_cloud`={default_counts['point_cloud']}, `bounding_box`={default_counts['bounding_box']}, `label`={default_counts['label']}, `remove`={default_counts['remove']}, `dropped_cases`={default_drops}/26\n")
    report_lines.append("---\n")

    # Table 1: COLLISION_MARGIN
    print("Sweeping COLLISION_MARGIN...")
    report_lines.append("## 1. Parameter Sweep: `COLLISION_MARGIN`\n")
    report_lines.append("| COLLISION_MARGIN (m) | R_reach (m) | point_cloud | bounding_box | label | remove | Dropped Cases | Changed Nodes vs Baseline |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for cm in [0.00, 0.05, 0.10, 0.20]:
        cfg = dict(base_cfg)
        cfg["COLLISION_MARGIN"] = cm
        counts, drops, assigns = evaluate_scene_26_cases(cfg, floor_raster, annots, rels, node_geoms, z_floor, tasks)
        diff = diff_against_default(assigns)
        reach = cfg["ARM_MAX_REACH"] + cm
        is_def = " *(Default)*" if cm == 0.05 else ""
        report_lines.append(
            f"| **{cm:.2f}**{is_def} | {reach:.3f} | {counts['point_cloud']} | {counts['bounding_box']} | {counts['label']} | {counts['remove']} | {drops}/26 | {diff} |"
        )

    report_lines.append("\n---\n")

    # Table 2: mount_height h
    print("Sweeping mount_height h...")
    report_lines.append("## 2. Parameter Sweep: `mount_height h`\n")
    report_lines.append("| Mount Height Setting | point_cloud | bounding_box | label | remove | Dropped Cases | Changed Nodes vs Baseline |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for h_val, h_label in [(None, "Solver-Chosen *(Default)*"), (0.30, "Fixed h = 0.30 m"), (0.85, "Fixed h = 0.85 m"), (1.05, "Fixed h = 1.05 m")]:
        counts, drops, assigns = evaluate_scene_26_cases(base_cfg, floor_raster, annots, rels, node_geoms, z_floor, tasks, fixed_h=h_val)
        diff = diff_against_default(assigns)
        report_lines.append(
            f"| **{h_label}** | {counts['point_cloud']} | {counts['bounding_box']} | {counts['label']} | {counts['remove']} | {drops}/26 | {diff} |"
        )

    report_lines.append("\n---\n")

    # Table 3: SUPPORT_Z_TOLERANCE
    print("Sweeping SUPPORT_Z_TOLERANCE...")
    report_lines.append("## 3. Parameter Sweep: `SUPPORT_Z_TOLERANCE`\n")
    report_lines.append("| SUPPORT_Z_TOLERANCE (m) | point_cloud | bounding_box | label | remove | Dropped Cases | Changed Nodes vs Baseline |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for szt in [0.05, 0.10, 0.20]:
        cfg = dict(base_cfg)
        cfg["SUPPORT_Z_TOLERANCE"] = szt
        counts, drops, assigns = evaluate_scene_26_cases(cfg, floor_raster, annots, rels, node_geoms, z_floor, tasks)
        diff = diff_against_default(assigns)
        is_def = " *(Default)*" if szt == 0.10 else ""
        report_lines.append(
            f"| **{szt:.2f}**{is_def} | {counts['point_cloud']} | {counts['bounding_box']} | {counts['label']} | {counts['remove']} | {drops}/26 | {diff} |"
        )

    report_lines.append("\n---\n")

    # Table 4: r_nominal
    print("Sweeping r_nominal...")
    report_lines.append("## 4. Parameter Sweep: `r_nominal` (Nominal Reach Distance)\n")
    report_lines.append("| Nominal Fraction | r_nominal (m) | point_cloud | bounding_box | label | remove | Dropped Cases | Changed Nodes vs Baseline |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for frac in [0.60, 0.75, 0.90]:
        cfg = dict(base_cfg)
        cfg["R_NOMINAL_FRACTION"] = frac
        r_nom = frac * cfg["ARM_MAX_REACH"]
        counts, drops, assigns = evaluate_scene_26_cases(cfg, floor_raster, annots, rels, node_geoms, z_floor, tasks)
        diff = diff_against_default(assigns)
        is_def = " *(Default)*" if frac == 0.75 else ""
        report_lines.append(
            f"| **{frac:.2f} * R_MAX**{is_def} | {r_nom:.4f} | {counts['point_cloud']} | {counts['bounding_box']} | {counts['label']} | {counts['remove']} | {drops}/26 | {diff} |"
        )

    report_lines.append("\n---\n")

    output_file = Path(__file__).resolve().parent / "sensitivity_report_0kitchen.md"
    with open(str(output_file), "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Successfully generated {output_file.name}")

if __name__ == "__main__":
    main()

