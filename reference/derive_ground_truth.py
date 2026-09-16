#!/usr/bin/env python3
"""
Deterministic Ground-Truth Derivation Script for Functional 3D Scene Graphs.

Supports two evaluation modes:
  --mode guideline : Reimplementation of annotation_policy.md's 2.0m Euclidean distance rule.
                     Validates data loading, subpart extraction, and geometry against T3 gold case.
  --mode kinematic : Physical derived ground truth based on continuous placement solver (A1)
                     and physical reachability levels (A2):
                       point_cloud  : Manipulated kinematic assembly (target body + articulated subparts)
                       bounding_box : OBB intersects reachable volume at CANONICAL placement
                       label        : OBB intersects reachable volume at SOME feasible placement (contingent)
                       remove       : No feasible placement for this task can ever reach it

THE RULE (from annotation guideline, quoted verbatim):
"1. The manipulated task object and its task-relevant functional subparts
   -> point_cloud
 2. Objects whose geometry matters for the interaction without being manipulated
   -> bounding_box. This covers:
     a. the object the task object rests on or is mounted to
     b. direct neighbours with which the robot arm could collide during manipulation
     c. the subparts of those neighbours where they protrude into the working space
 3. Objects that neither carry nor obstruct the interaction -> label (kept as context)
 4. The subparts of those context objects -> remove"
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import scipy.ndimage as ndi
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ==============================================================================
# CONFIG BLOCK
# Every numeric constant lives here, named, commented with source, overridable.
# ==============================================================================
CONFIG = {
    # Robot manipulator specification (Piper 6-DOF lightweight arm)
    # Source: Piper robot hardware specification (reach = 0.626 m)
    "ARM_MAX_REACH": 0.626,

    # Inner dead-zone radius of the arm
    # Source: Piper URDF piper_description.urdf (L2 = 0.28503 m, L3 = 0.25075 m,
    # joint 3 limit -2.967 rad allows links 2 and 3 to fold back to |L2 - L3| = 0.034 m)
    "ARM_MIN_REACH": 0.034,

    # Reach margin deducted from ARM_MAX_REACH to establish effective reach
    # Source: Conservative reach margin ensuring base stance sits within manipulator reach envelope.
    # NOTE: The placement annulus is conservative-small (R_eff = 0.626 - 0.05 = 0.576 m),
    # while the collision envelope is conservative-large (R_reach = 0.626 + 0.05 = 0.676 m).
    # A node can therefore be classified as bounding_box at a radius the solver will not place a base for.
    "REACH_MARGIN": 0.05,

    # Collision margin defining "could collide" buffer around arm reachable volume
    # Source: Standard robotics safety clearance margin (0.05 m)
    "COLLISION_MARGIN": 0.05,

    # Robot base footprint width (0.30 m square)
    # Source: Nominal mobile base footprint width (0.30 m)
    "BASE_FOOTPRINT_WIDTH": 0.30,

    # Additional base clearance margin beyond base footprint half-width
    # Source: Stand-off clearance matching raster grid resolution (0.02 m)
    "BASE_CLEARANCE": 0.02,

    # Rasterization resolution for 2D free-floor grid (0.02 m = 2 cm)
    # Source: 2 cm spatial discretization resolution
    "GRID_RESOLUTION": 0.02,

    # Nominal reach fraction for canonical base placement selection
    # Source: Nominal stance distance preferred at 75% of maximum reach (r_nominal = 0.75 * ARM_MAX_REACH)
    "R_NOMINAL_FRACTION": 0.75,

    # Admissible physical mount height sweep range above floor
    # Source: run_manipulation_geometry_experiment.py
    "MOUNT_HEIGHT_MIN": 0.30,
    "MOUNT_HEIGHT_MAX": 1.40,
    "MOUNT_HEIGHT_STEP": 0.01,  # 1 cm sweep resolution

    # Geometric support detection tolerances (Rule 2a)
    # Source: Physical contact threshold accounting for scan noise / surface thickness
    "SUPPORT_Z_TOLERANCE": 0.10,

    # Minimum fractional overlap in XY plane for vertical support (Rule 2a)
    # Source: Conservative threshold for stable vertical resting contact
    "SUPPORT_XY_OVERLAP_MIN": 0.10,

    # Guideline mode distance threshold (used ONLY in --mode guideline)
    # Source: annotation_policy.md Section 2 (2.0 m Euclidean distance sphere)
    "GUIDELINE_DISTANCE_THRESHOLD": 2.0,
}

# Exhaustive enumeration of all accepted mechanical affordance verbs across FunGraph3D
# Source: FunGraph3D.relations.json inspection across all 14 scenes
ACCEPTED_MECHANICAL_VERBS = {
    "control",
    "control the water flow",
    "control to drain the water flow",
    "control, turn on or turn off",
    "operate or adjust the flow or setting",
    "press or rotate to control the drain water flow",
    "press or rotate to control the water flow",
    "press to flush",
    "press to open or close, or to adjust the setting",
    "press to open or close, or to adjust the wind flow",
    "press to start and operate",
    "pull to open or close",
    "pull to open or close a cabinet or drawer",
    "pull to open or close a drawer",
    "push or press to open",
    "rotate to adjust temperature or setting, open or close",
    "rotate to adjust the setting",
    "rotate to adjust the setting or temperature, or to open or close",
    "rotate to adjust the temperature",
    "rotate to open or close",
    "rotate to open or close, adjust the setting",
    "rotate to open or close, or to adjust the setting and temperature",
    "rotate to open or close, or to adjust the setting, time and temperature",
    "rotate to open or close, or to adjust the temperature and setting",
}

# Non-mechanical edge descriptions that do NOT establish subpart relations
KNOWN_NON_MECHANICAL_VERBS = {
    "provide power",
}

# Storage-related categories recognized by annotation_policy.md
GUIDELINE_STORAGE_CATEGORIES = {
    "kitchen cabinet", "fridge", "dishwasher", "wardrobe",
    "dresser", "desk drawer", "vanity"
}


# ==============================================================================
# DATA LOADERS & PREPROCESSING
# ==============================================================================

def load_dataset_annotations(dataset_dir: str, scene_id: str) -> List[Dict[str, Any]]:
    """Load annotations for a scene from FunGraph3D.annotations.json."""
    if not os.path.exists(dataset_dir):
        alt = REPO_ROOT / dataset_dir
        if alt.exists():
            dataset_dir = str(alt)
    ann_path = os.path.join(dataset_dir, "FunGraph3D.annotations.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations file not found: {ann_path}")
    with open(ann_path, "r") as f:
        data = json.load(f)
    scene_nodes = [a for a in data if a.get("scene_id") == scene_id]
    if not scene_nodes:
        raise ValueError(f"No annotations found for scene '{scene_id}' in {ann_path}")
    return scene_nodes


def load_dataset_relations(dataset_dir: str, scene_id: str) -> List[Dict[str, Any]]:
    """Load relations for a scene from FunGraph3D.relations.json."""
    if not os.path.exists(dataset_dir):
        alt = REPO_ROOT / dataset_dir
        if alt.exists():
            dataset_dir = str(alt)
    rel_path = os.path.join(dataset_dir, "FunGraph3D.relations.json")
    if not os.path.exists(rel_path):
        raise FileNotFoundError(f"Relations file not found: {rel_path}")
    with open(rel_path, "r") as f:
        data = json.load(f)
    return [r for r in data if r.get("scene_id") == scene_id]


def load_node_geometries(geom_path: str, scene_id: str) -> Dict[str, Dict[str, Any]]:
    """Load precomputed node geometries (centroids, bounding box min/max, vertex counts)."""
    if not os.path.exists(geom_path):
        alt = REPO_ROOT / "data" / "geometry" / os.path.basename(geom_path)
        if alt.exists():
            geom_path = str(alt)
        elif (REPO_ROOT / geom_path).exists():
            geom_path = str(REPO_ROOT / geom_path)
        else:
            raise FileNotFoundError(f"Node geometry file not found: {geom_path}")
    with open(geom_path, "r") as f:
        data = json.load(f)
    scene_geoms = {k: v for k, v in data.items() if v.get("scene") == scene_id}
    return scene_geoms


def load_floor_elevation(affordance_path: str, scene_id: str) -> float:
    """Load fitted floor elevation Z_floor for the scene."""
    if not os.path.exists(affordance_path):
        alt = REPO_ROOT / "data" / "geometry" / os.path.basename(affordance_path)
        if alt.exists():
            affordance_path = str(alt)
        elif (REPO_ROOT / affordance_path).exists():
            affordance_path = str(REPO_ROOT / affordance_path)
        else:
            raise FileNotFoundError(f"Affordance heights file not found: {affordance_path}")
    with open(affordance_path, "r") as f:
        data = json.load(f)
    floor_levels = data.get("floor_levels", {})
    if scene_id not in floor_levels:
        raise ValueError(f"Floor level not found for scene '{scene_id}'")
    return float(floor_levels[scene_id])


def extract_subpart_relationships(
    relations: List[Dict[str, Any]]
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, List[str]]]:
    """
    Extract parent-subpart relationships from relation graph.
    RAISES ValueError if an unrecognised relation verb is encountered.
    """
    child_to_parent: Dict[str, Tuple[str, str]] = {}
    parent_to_children: Dict[str, List[str]] = {}

    for r in relations:
        child_id = r["first_node_annot_id"]
        parent_id = r["second_node_annot_id"]
        desc = r.get("description", "")

        if desc in ACCEPTED_MECHANICAL_VERBS:
            child_to_parent[child_id] = (parent_id, desc)
            parent_to_children.setdefault(parent_id, []).append(child_id)
        elif desc in KNOWN_NON_MECHANICAL_VERBS:
            continue
        else:
            raise ValueError(
                f"UNRECOGNIZED RELATION VERB encountered in relations: '{desc}' "
                f"between {child_id} and {parent_id}. "
                f"Must be enumerated in ACCEPTED_MECHANICAL_VERBS or KNOWN_NON_MECHANICAL_VERBS."
            )

    return child_to_parent, parent_to_children


def point_to_box_distance(point: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """Compute exact Euclidean distance from a 3D point to an axis-aligned box [bmin, bmax]."""
    clamped = np.clip(point, bmin, bmax)
    return float(np.linalg.norm(point - clamped))


# ==============================================================================
# A1: INVERTED CONTINUOUS PLACEMENT SOLVER
# ==============================================================================

class SceneFloorRaster:
    """Computes and caches the 2D free-floor raster at 2 cm resolution."""
    def __init__(self, ply_path: str, z_floor: float, node_geoms: Dict[str, Dict[str, Any]], res: float = 0.02):
        self.res = res
        self.z_floor = z_floor
        mesh = trimesh.load(ply_path)
        pts = mesh.vertices
        z_rect = pts[:, 2] - z_floor

        self.x_min, self.y_min = np.floor(pts[:, :2].min(axis=0) / res) * res - 0.1
        self.x_max, self.y_max = np.ceil(pts[:, :2].max(axis=0) / res) * res + 0.1
        self.nx = int(round((self.x_max - self.x_min) / res))
        self.ny = int(round((self.y_max - self.y_min) / res))

        # Ground points mask
        ground_pts = pts[z_rect < 0.05]
        gx = np.clip(np.floor((ground_pts[:, 0] - self.x_min) / res).astype(int), 0, self.nx - 1)
        gy = np.clip(np.floor((ground_pts[:, 1] - self.y_min) / res).astype(int), 0, self.ny - 1)
        floor_grid = np.zeros((self.nx, self.ny), dtype=bool)
        floor_grid[gx, gy] = True
        self.floor_grid = ndi.binary_closing(floor_grid, structure=np.ones((5, 5), dtype=bool))

        # Obstacle points (0.05 <= z <= 1.80)
        obs_pts = pts[(z_rect >= 0.05) & (z_rect <= 1.80)]
        ox = np.clip(np.floor((obs_pts[:, 0] - self.x_min) / res).astype(int), 0, self.nx - 1)
        oy = np.clip(np.floor((obs_pts[:, 1] - self.y_min) / res).astype(int), 0, self.ny - 1)
        obs_grid = np.zeros((self.nx, self.ny), dtype=bool)
        obs_grid[ox, oy] = True

        # Node OBB projections
        for k, v in node_geoms.items():
            if v["max"][2] - z_floor >= 0.05:
                bx0 = max(0, int(np.floor((v["min"][0] - self.x_min) / res)))
                bx1 = min(self.nx - 1, int(np.ceil((v["max"][0] - self.x_min) / res)))
                by0 = max(0, int(np.floor((v["min"][1] - self.y_min) / res)))
                by1 = min(self.ny - 1, int(np.ceil((v["max"][1] - self.y_min) / res)))
                obs_grid[bx0:bx1+1, by0:by1+1] = True

        self.dist_to_obs = ndi.distance_transform_edt(~obs_grid) * res

        grid_x = self.x_min + (np.arange(self.nx) + 0.5) * res
        grid_y = self.y_min + (np.arange(self.ny) + 0.5) * res
        self.GX, self.GY = np.meshgrid(grid_x, grid_y, indexing="ij")

    def get_free_floor(self, w_base: float = 0.30, base_clearance: float = 0.02) -> np.ndarray:
        required_clearance = (w_base / 2.0) + base_clearance
        return self.floor_grid & (self.dist_to_obs >= required_clearance)


def solve_feasible_base_placements(
    affordance_p: np.ndarray,
    floor_raster: SceneFloorRaster,
    z_floor: float,
    config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Solve feasible base placements for a target affordance point p using Annulus search (A1).
    Returns (feasible_placements, drop_reason).
    drop_reason is None if placed, '(i) vertically infeasible', or '(ii) blocked'.
    """
    arm_reach = config["ARM_MAX_REACH"]
    reach_margin = config.get("REACH_MARGIN", 0.0)
    r_eff = arm_reach - reach_margin
    r_min = config.get("ARM_MIN_REACH", 0.0)
    r_nominal = config.get("R_NOMINAL_FRACTION", 0.75) * arm_reach

    h_min_phys = config["MOUNT_HEIGHT_MIN"]
    h_max_phys = config["MOUNT_HEIGHT_MAX"]
    h_step = config.get("MOUNT_HEIGHT_STEP", 0.01)

    h_min_kin = (affordance_p[2] - z_floor) - r_eff
    h_max_kin = (affordance_p[2] - z_floor) + r_eff
    h_start = max(h_min_phys, h_min_kin)
    h_end = min(h_max_phys, h_max_kin)

    if h_start > h_end:
        return [], "(i) vertically infeasible"

    free_floor = floor_raster.get_free_floor(
        config["BASE_FOOTPRINT_WIDTH"], config["BASE_CLEARANCE"]
    )
    dist_xy = np.sqrt((floor_raster.GX - affordance_p[0])**2 + (floor_raster.GY - affordance_p[1])**2)

    h_steps = np.arange(round(h_start, 2), round(h_end, 2) + 0.005, h_step)
    feasible_placements: List[Dict[str, Any]] = []

    for h in h_steps:
        z_s = z_floor + h
        dz = affordance_p[2] - z_s
        if abs(dz) > r_eff:
            continue
        rho_outer = np.sqrt(r_eff**2 - dz**2)
        rho_inner = np.sqrt(max(0.0, r_min**2 - dz**2))

        in_annulus = (dist_xy >= rho_inner) & (dist_xy <= rho_outer)
        feasible = in_annulus & free_floor

        if np.any(feasible):
            fx_idx, fy_idx = np.where(feasible)
            for i, j in zip(fx_idx, fy_idx):
                bx = floor_raster.GX[i, j]
                by = floor_raster.GY[i, j]
                clearance = floor_raster.dist_to_obs[i, j]
                d_shoulder = np.sqrt((bx - affordance_p[0])**2 + (by - affordance_p[1])**2 + dz**2)
                cost1 = abs(d_shoulder - r_nominal)
                yaw_rad = math.atan2(affordance_p[1] - by, affordance_p[0] - bx)
                yaw_deg = (math.degrees(yaw_rad) % 360)

                feasible_placements.append({
                    "h": round(float(h), 4),
                    "base_x": round(float(bx), 4),
                    "base_y": round(float(by), 4),
                    "shoulder_z": round(float(z_s), 4),
                    "base_yaw_deg": round(float(yaw_deg), 2),
                    "base_yaw_rad": yaw_rad,
                    "cost1": cost1,
                    "clearance": clearance,
                    "d_shoulder": d_shoulder,
                })

    if not feasible_placements:
        return [], "(ii) blocked"

    # Canonical selection lexicographic:
    # 1. minimise |dist(p, shoulder) - r_nominal|
    # 2. maximise clearance to nearest obstacle (-clearance)
    # 3. smallest h, then smallest yaw
    feasible_placements.sort(
        key=lambda c: (round(c["cost1"], 3), -round(c["clearance"], 3), c["h"], c["base_yaw_deg"])
    )

    return feasible_placements, None


# ==============================================================================
# MODE IMPLEMENTATIONS: GUIDELINE & KINEMATIC
# ==============================================================================

def derive_ground_truth_guideline(
    scene_id: str,
    task: str,
    target_object_id: str,
    annotations: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    node_geoms: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Faithful reimplementation of annotation_policy.md's 2.0m Euclidean distance rule.
    Anchors on target OBJECT centroid.
    Must reproduce T3 exactly (4 pc / 16 bbox / 7 label / 12 remove on 0kitchen microwave).
    """
    child_to_parent, parent_to_children = extract_subpart_relationships(relations)
    d_threshold = config.get("GUIDELINE_DISTANCE_THRESHOLD", 2.0)

    if target_object_id not in node_geoms:
        raise ValueError(f"Target object '{target_object_id}' not found in node geometry.")

    t_centroid = np.array(node_geoms[target_object_id]["centroid"])
    assignments: Dict[str, Dict[str, Any]] = {}

    # 1. Target and its mechanical controls -> point_cloud
    assignments[target_object_id] = {
        "label": node_geoms[target_object_id]["label"],
        "detail": "point_cloud",
        "rule": "1",
        "evidence": {"is_target": True, "distance_to_target": 0.0}
    }

    for sub_id in parent_to_children.get(target_object_id, []):
        sub_g = node_geoms[sub_id]
        dist = float(np.linalg.norm(np.array(sub_g["centroid"]) - t_centroid))
        assignments[sub_id] = {
            "label": sub_g["label"],
            "detail": "point_cloud",
            "rule": "1",
            "evidence": {"subpart_of": target_object_id, "distance_to_target": round(dist, 4)}
        }

    # Find nodes with active provide power relations
    power_nodes = set()
    for r in relations:
        if r.get("description") == "provide power":
            power_nodes.add(r["first_node_annot_id"])

    # 2. Assign major objects (non-subparts)
    for ann in annotations:
        nid = ann["annot_id"]
        if nid in assignments or nid in child_to_parent:
            continue
        g = node_geoms[nid]
        lbl = g["label"]
        dist = float(np.linalg.norm(np.array(g["centroid"]) - t_centroid))

        # Check power component
        if nid in power_nodes:
            assignments[nid] = {
                "label": lbl,
                "detail": "remove",
                "rule": "4",
                "evidence": {"active_power_source": True, "distance_to_target": round(dist, 4)}
            }
            continue

        # Switches and panels
        if any(w in lbl.lower() for w in ["switch", "panel"]):
            assignments[nid] = {
                "label": lbl,
                "detail": "remove",
                "rule": "4",
                "evidence": {"switch_or_panel": True, "distance_to_target": round(dist, 4)}
            }
            continue

        # Storage vs Context fixtures
        is_storage = lbl in GUIDELINE_STORAGE_CATEGORIES

        if is_storage:
            if dist <= d_threshold:
                assignments[nid] = {
                    "label": lbl,
                    "detail": "bounding_box",
                    "rule": "2",
                    "evidence": {"storage_near": True, "distance_to_target": round(dist, 4)}
                }
            else:
                assignments[nid] = {
                    "label": lbl,
                    "detail": "label",
                    "rule": "3",
                    "evidence": {"storage_distant_demoted": True, "distance_to_target": round(dist, 4)}
                }
        else:
            if dist <= d_threshold:
                assignments[nid] = {
                    "label": lbl,
                    "detail": "label",
                    "rule": "3",
                    "evidence": {"fixture_near": True, "distance_to_target": round(dist, 4)}
                }
            else:
                assignments[nid] = {
                    "label": lbl,
                    "detail": "remove",
                    "rule": "4",
                    "evidence": {"fixture_distant_demoted": True, "distance_to_target": round(dist, 4)}
                }

    # 3. Assign remaining subparts based on parent role
    for ann in annotations:
        nid = ann["annot_id"]
        if nid in assignments:
            continue
        if nid in child_to_parent:
            p_id = child_to_parent[nid][0]
            g = node_geoms[nid]
            dist = float(np.linalg.norm(np.array(g["centroid"]) - t_centroid))
            p_detail = assignments[p_id]["detail"]

            if p_detail == "bounding_box" and dist <= d_threshold:
                assignments[nid] = {
                    "label": g["label"],
                    "detail": "bounding_box",
                    "rule": "2c",
                    "evidence": {"parent_id": p_id, "parent_level": "bounding_box", "distance_to_target": round(dist, 4)}
                }
            else:
                assignments[nid] = {
                    "label": g["label"],
                    "detail": "remove",
                    "rule": "4",
                    "evidence": {"parent_id": p_id, "parent_level": p_detail, "distance_to_target": round(dist, 4)}
                }

    counts = {
        "point_cloud": sum(1 for v in assignments.values() if v["detail"] == "point_cloud"),
        "bounding_box": sum(1 for v in assignments.values() if v["detail"] == "bounding_box"),
        "label": sum(1 for v in assignments.values() if v["detail"] == "label"),
        "remove": sum(1 for v in assignments.values() if v["detail"] == "remove"),
    }

    return {
        "scene_id": scene_id,
        "task": task,
        "mode": "guideline",
        "target_node_id": target_object_id,
        "critical_target": target_object_id,
        "counts": counts,
        "resolved_config": dict(config),
        "detail_assignment": assignments,
    }


def resolve_task_targets(task_dict: Dict[str, Any]) -> Tuple[str, str]:
    """
    Hardened target object and affordance resolver (F2).
    Resolves target_obj_id and target_aff_id deterministically.
    Raises ValueError if target_object or affordance cannot be resolved.
    Never defaults silently.
    """
    fn = task_dict.get("filename", "")
    tos = task_dict.get("target_objects", [])
    if not tos:
        raise ValueError(f"Task '{fn}' has no target_objects defined.")

    if len(tos) > 1:
        # Disambiguate multiple target objects
        if "microwave" in fn.lower():
            matched = [to for to in tos if "microwave" in to["label"].lower()]
        elif "oven" in fn.lower() or "hungry_2" in fn.lower() or "hungry_3" in fn.lower():
            matched = [to for to in tos if to["label"].lower() == "oven"]
        elif "bored" in fn.lower():
            matched = [to for to in tos if "television" in to["label"].lower() or "tv" in to["label"].lower()]
        else:
            matched = [to for to in tos if any(w in fn.lower() for w in to["label"].lower().split())]
        if not matched:
            raise ValueError(f"Task '{fn}' has ambiguous target objects: {tos}")
        target_obj = matched[0]
    else:
        target_obj = tos[0]

    target_obj_id = target_obj["node_id"]

    affs = task_dict.get("affordance_nodes", [])
    if affs:
        matching_affs = [
            a for a in affs
            if a.get("target_node_id") == target_obj_id or a["node_id"] == target_obj_id
        ]
        if not matching_affs:
            matching_affs = [a for a in affs if "remote" in a.get("label", "").lower()] or affs
        handles = [a for a in matching_affs if "handle" in a.get("label", "").lower() or "pull" in a.get("action", "").lower() or "remote" in a.get("label", "").lower()]
        selected_aff = handles[0] if handles else matching_affs[0]
        target_aff_id = selected_aff["node_id"]
    else:
        # Direct manipulable objects (e.g. kettle, dishwasher)
        target_aff_id = target_obj_id

    if not target_aff_id:
        raise ValueError(f"Task '{fn}' could not resolve target_aff_id.")

    return target_obj_id, target_aff_id


def derive_ground_truth_kinematic(
    scene_id: str,
    task: str,
    target_object_id: str,
    affordance_id: str,
    floor_raster: SceneFloorRaster,
    annotations: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    node_geoms: Dict[str, Dict[str, Any]],
    z_floor: float,
    config: Dict[str, Any],
    custom_base_pose: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Physical derived ground truth (A1 + A2).
    Anchors on target affordance. Evaluates canonical (bounding_box), union (label), and unreachable (remove).
    """
    child_to_parent, parent_to_children = extract_subpart_relationships(relations)
    r_reach = config["ARM_MAX_REACH"] + config["COLLISION_MARGIN"]

    if affordance_id not in node_geoms:
        raise ValueError(f"Affordance node '{affordance_id}' not found in node geometry.")
    p_aff = np.array(node_geoms[affordance_id]["centroid"])

    # 1. Solve base placements
    if custom_base_pose is not None:
        can = dict(custom_base_pose)
        can_shoulder = np.array([can["base_x"], can["base_y"], z_floor + can["h"]])
        all_shoulders = np.array([can_shoulder])
        feasible_placements = [can]
        drop_reason = None
    else:
        feasible_placements, drop_reason = solve_feasible_base_placements(p_aff, floor_raster, z_floor, config)
        if drop_reason is not None:
            # Dropped case: assign kinematic assembly to point_cloud, everything else remove
            kinematic_assembly = {target_object_id}
            for r in relations:
                if r["second_node_annot_id"] == target_object_id and r.get("description") in ACCEPTED_MECHANICAL_VERBS:
                    kinematic_assembly.add(r["first_node_annot_id"])

            assignments: Dict[str, Dict[str, Any]] = {}
            for ann in annotations:
                nid = ann["annot_id"]
                lbl = node_geoms[nid]["label"]
                if nid in kinematic_assembly:
                    assignments[nid] = {
                        "label": lbl, "detail": "point_cloud", "rule": "1",
                        "evidence": {"is_kinematic_assembly": True, "drop_reason": drop_reason}
                    }
                else:
                    assignments[nid] = {
                        "label": lbl, "detail": "remove", "rule": "4",
                        "evidence": {"unreachable_dropped_task": True, "drop_reason": drop_reason}
                    }

            counts = {
                "point_cloud": len(kinematic_assembly), "bounding_box": 0,
                "label": 0, "remove": len(annotations) - len(kinematic_assembly)
            }
            return {
                "scene_id": scene_id,
                "task": task,
                "mode": "kinematic",
                "target_node_id": target_object_id,
                "critical_target": target_object_id,
                "target_affordance_id": affordance_id,
                "dropped": True,
                "drop_reason": drop_reason,
                "feasible_placements_count": 0,
                "counts": counts,
                "canonical_placement": None,
                "resolved_config": dict(config),
                "detail_assignment": assignments,
            }

        can = feasible_placements[0]
        can_shoulder = np.array([can["base_x"], can["base_y"], can["shoulder_z"]])
        all_shoulders = np.array([[x["base_x"], x["base_y"], x["shoulder_z"]] for x in feasible_placements])

    # 2. Rule 1: Physical Kinematic Assembly
    kinematic_assembly = {target_object_id}
    for r in relations:
        if r["second_node_annot_id"] == target_object_id and r.get("description") in ACCEPTED_MECHANICAL_VERBS:
            kinematic_assembly.add(r["first_node_annot_id"])

    assignments = {}
    for sub_id in kinematic_assembly:
        assignments[sub_id] = {
            "label": node_geoms[sub_id]["label"],
            "detail": "point_cloud",
            "rule": "1",
            "evidence": {
                "is_kinematic_assembly": True,
                "target_object_id": target_object_id,
            }
        }

    # 3. Canonical and Union Reachability
    rule_2b_audit = []
    for ann in annotations:
        nid = ann["annot_id"]
        if nid in kinematic_assembly:
            continue
        g = node_geoms[nid]
        bmin = np.array(g["min"])
        bmax = np.array(g["max"])
        extents = bmax - bmin
        vol = float(extents[0] * extents[1] * extents[2])

        # Canonical check
        d_can = point_to_box_distance(can_shoulder, bmin, bmax)
        if d_can <= r_reach:
            pen = round(r_reach - d_can, 4)
            assignments[nid] = {
                "label": g["label"],
                "detail": "bounding_box",
                "rule": "2b",
                "evidence": {
                    "obb_workspace_distance": round(d_can, 4),
                    "penetration_depth": pen,
                    "obb_side_lengths": [round(float(x), 3) for x in extents],
                    "obb_volume": round(vol, 4),
                    "workspace_radius": round(r_reach, 4),
                }
            }
            rule_2b_audit.append({
                "node_id": nid,
                "label": g["label"],
                "extents": [round(float(x), 3) for x in extents],
                "volume": round(vol, 4),
                "penetration": pen,
                "distance": round(d_can, 4),
            })
            continue

        # Union check across all feasible shoulders
        clamped_all = np.clip(all_shoulders, bmin, bmax)
        d_all = np.linalg.norm(all_shoulders - clamped_all, axis=1)
        min_d_union = float(np.min(d_all))

        if min_d_union <= r_reach:
            assignments[nid] = {
                "label": g["label"],
                "detail": "label",
                "rule": "3",
                "evidence": {
                    "contingent_obstacle": True,
                    "min_distance_to_union_workspace": round(min_d_union, 4),
                    "canonical_distance": round(d_can, 4),
                    "workspace_radius": round(r_reach, 4),
                }
            }
        else:
            assignments[nid] = {
                "label": g["label"],
                "detail": "remove",
                "rule": "4",
                "evidence": {
                    "unreachable_across_all_placements": True,
                    "min_distance_to_union_workspace": round(min_d_union, 4),
                    "workspace_radius": round(r_reach, 4),
                }
            }

    counts = {
        "point_cloud": sum(1 for v in assignments.values() if v["detail"] == "point_cloud"),
        "bounding_box": sum(1 for v in assignments.values() if v["detail"] == "bounding_box"),
        "label": sum(1 for v in assignments.values() if v["detail"] == "label"),
        "remove": sum(1 for v in assignments.values() if v["detail"] == "remove"),
    }

    resolved_cfg = dict(config)
    resolved_cfg.update({
        "canonical_base_x": can["base_x"],
        "canonical_base_y": can["base_y"],
        "canonical_mount_height_h": can["h"],
        "canonical_yaw_deg": can["base_yaw_deg"],
        "z_floor": z_floor,
        "feasible_placements_count": len(feasible_placements),
        "workspace_radius": r_reach,
    })

    return {
        "scene_id": scene_id,
        "task": task,
        "mode": "kinematic",
        "target_node_id": target_object_id,
        "critical_target": target_object_id,
        "target_affordance_id": affordance_id,
        "dropped": False,
        "drop_reason": None,
        "feasible_placements_count": len(feasible_placements),
        "all_feasible_placements": feasible_placements,
        "counts": counts,
        "canonical_placement": can,
        "rule_2b_audit": rule_2b_audit,
        "resolved_config": resolved_cfg,
        "detail_assignment": assignments,
    }


# ==============================================================================
# MAIN CLI ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deterministic Ground-Truth Derivation Script")
    parser.add_argument("--scene", default="0kitchen", help="Scene ID")
    parser.add_argument("--task", default="open the microwave oven", help="Task string")
    parser.add_argument("--candidate-target", default=None, help="Candidate target object UUID")
    parser.add_argument("--affordance", default=None, help="Candidate affordance UUID")
    parser.add_argument("--mode", choices=["kinematic", "guideline"], default="kinematic", help="Derivation mode")
    parser.add_argument("--base-x", type=float, default=None, help="Explicit robot base X")
    parser.add_argument("--base-y", type=float, default=None, help="Explicit robot base Y")
    parser.add_argument("--base-yaw", type=float, default=None, help="Explicit robot base Yaw (deg)")
    parser.add_argument("--mount-height", type=float, default=None, help="Explicit arm mount height h above floor")
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "dataset/FunGraph3D"), help="Path to FunGraph3D")
    parser.add_argument("--node-geom", default=str(REPO_ROOT / "data/geometry/node_geom.json"), help="Path to node_geom.json")
    parser.add_argument("--affordance-heights", default=str(REPO_ROOT / "data/geometry/affordance_heights.json"), help="Path to affordance_heights.json")

    args = parser.parse_args()

    scene_id = args.scene
    task = args.task

    annots = load_dataset_annotations(args.dataset_dir, scene_id)
    rels = load_dataset_relations(args.dataset_dir, scene_id)
    node_geoms = load_node_geometries(args.node_geom, scene_id)
    z_floor = load_floor_elevation(args.affordance_heights, scene_id)

    target_obj_id = args.candidate_target
    affordance_id = args.affordance
    if not target_obj_id:
        raise ValueError("Must provide --candidate-target")
    if args.mode == "kinematic" and not affordance_id:
        raise ValueError("Must provide --affordance for kinematic mode")

    if args.mode == "guideline":
        res = derive_ground_truth_guideline(
            scene_id=scene_id,
            task=task,
            target_object_id=target_obj_id,
            annotations=annots,
            relations=rels,
            node_geoms=node_geoms,
            config=CONFIG,
        )
    else:
        ply_path = os.path.join(args.dataset_dir, scene_id, f"{scene_id}.ply")
        floor_raster = SceneFloorRaster(ply_path, z_floor, node_geoms, res=CONFIG["GRID_RESOLUTION"])

        custom_pose = None
        if args.base_x is not None and args.base_y is not None and args.mount_height is not None:
            custom_pose = {
                "base_x": args.base_x,
                "base_y": args.base_y,
                "h": args.mount_height,
                "base_yaw_deg": args.base_yaw if args.base_yaw is not None else 0.0,
            }

        res = derive_ground_truth_kinematic(
            scene_id=scene_id,
            task=task,
            target_object_id=target_obj_id,
            affordance_id=affordance_id,
            floor_raster=floor_raster,
            annotations=annots,
            relations=rels,
            node_geoms=node_geoms,
            z_floor=z_floor,
            config=CONFIG,
            custom_base_pose=custom_pose,
        )

    print(json.dumps({
        "scene_id": res["scene_id"],
        "task": res["task"],
        "mode": res["mode"],
        "target_node_id": res["target_node_id"],
        "counts": res["counts"],
        "canonical_placement": res.get("canonical_placement"),
        "rule_2b_audit": res.get("rule_2b_audit", []),
    }, indent=2))


if __name__ == "__main__":
    main()
