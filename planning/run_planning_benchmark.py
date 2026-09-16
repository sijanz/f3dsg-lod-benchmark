#!/usr/bin/env python3
"""
run_planning_benchmark.py — Motion Planning Benchmark Batch Runner.

Executes MoveIt 2 motion planning evaluation across benchmark cases and policy arms.
Evaluates reachability, planning feasibility, path validity, and collision clearance
over 3,720 trials (155 cases x 8 policy arms x 3 repeats).
"""

import argparse
import datetime
import hashlib
import json
import math
import mmap
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ROS 2 & MoveIt imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
from moveit_msgs.msg import PlanningScene, LinkPadding, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity, GetPlanningScene

# Workspace paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "dataset" / "FunGraph3D"
HEIGHTS_PATH = REPO_ROOT / "data" / "affordance_heights.json"
GEOM_PATH = REPO_ROOT / "data" / "node_geom.json"
KINEMATICS_PATH = REPO_ROOT / "data" / "ground_truth_kinematics_reselected.json"
DEFAULTS_PATH = REPO_ROOT / "evaluation" / "all_10_arms_evaluation.json"
SRDF_PATH = Path(os.environ.get("PIPER_SRDF_PATH", REPO_ROOT / "piper_ros/src/piper_moveit/piper_with_gripper_moveit/config/piper.srdf"))

sys.path.insert(0, str(REPO_ROOT))

# Pinned simulation modules
from planning.support.piper_moveit_planner import PiperMoveItPlanner
from planning.support.scene_builder import MoveItSceneBuilder, rectify_node_geometries
from reference.derive_ground_truth import load_node_geometries, load_dataset_annotations, load_floor_elevation
from planning.support.kdl_utils import build_kdl_chain
from planning.support.octomap_verifier import get_resident_octree_leaf_count

# Expected SHA-256 Hashes for Registry Assertions
EXPECTED_KINEMATICS_HASH = "36ca56e562033f392e07edf16d1491adc0371d7ad70c270dcfc7a84ba4fa8687"
EXPECTED_DEFAULTS_HASH = "b6517b1abf96d8660ce6b69fa1c28b601970784fc74509ea5756d212f9e44b97"
BUILD_HASH_SCENE_BUILDER = "27e24e83ad1c928dcd2aeca2aea94d8c56b3d7618e377941d60a5fef075db0ef"
BUILD_HASH_OCTOMAP_SO = "ed723984e73f8cba17ee6dc6aec2b228875823eafc35713ef0474a002df36697"
BUILD_HASH_PIPER_SRDF = "ead07feeecc122c9902436de448d58a43e6faba37b7c484c7dfd55bea9821156"

# Pinned 8 Evaluation Arms
PINNED_EVAL_ARMS = [
    "identity",
    "label_radius",
    "combined_quantile_adaptive",
    "affordance_inherit",
    "label_inherit",
    "label_only",
    "combined_inherit",
    "information_bottleneck"
]

ROBOT_LINKS = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]
ALL_ROBOT_LINKS = {"base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base", "link7", "link8"}


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_move_group_pid() -> int:
    try:
        out = subprocess.check_output(["pgrep", "-x", "move_group"]).decode().strip().split()
        return int(out[0]) if out else -1
    except Exception:
        return -1


def get_rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/statm") as f:
            pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


def restart_move_group_process(timeout_sec: float = 35.0) -> Tuple[int, float]:
    """Execute planned restart of move_group with full capability readiness synchronization."""
    t0 = time.time()
    print("[PLANNED RESTART] Terminating existing move_group process...", flush=True)
    subprocess.run(["pkill", "-9", "-x", "move_group"])
    time.sleep(2.0)
    launch_cmd = [
        "ros2", "launch",
        str(REPO_ROOT / "planning/launch/piper_sim_moveit.launch.py")
    ]
    log_path = "/tmp/moveit_phaseZ_batch.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        launch_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    new_pid = -1
    t_deadline = t0 + timeout_sec
    while time.time() < t_deadline:
        time.sleep(0.5)
        new_pid = get_move_group_pid()
        if new_pid > 0:
            break
    if new_pid <= 0:
        raise RuntimeError("Failed to launch move_group process during restart!")

    # Wait for MoveGroup full initialization ("You can start planning now!")
    ready = False
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path, "r", errors="ignore") as f:
                content = f.read()
                if "You can start planning now!" in content:
                    ready = True
                    break
        time.sleep(0.5)

    if not ready:
        raise RuntimeError("MoveGroup did not report readiness within timeout!")

    time.sleep(1.0)
    t_elapsed = time.time() - t0
    rss = get_rss_mb(new_pid)
    print(f"[PLANNED RESTART COMPLETE] New PID: {new_pid} | Ready in {t_elapsed:.3f} s | Initial RSS: {rss:.2f} MB", flush=True)
    return new_pid, t_elapsed


# Raw PLY loading with floor elevation rectification
def load_ply_points(dataset_path: str, scene_id: str) -> np.ndarray:
    """
    Direct mmap reader for raw PLY points.
    Does NOT call load_ply_and_rectify and does NOT fit a floor plane from PLY.
    Floor elevation comes exclusively from affordance_heights.json.
    """
    ply_path = os.path.join(dataset_path, scene_id, f"{scene_id}.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing PLY file: {ply_path}")
    with open(ply_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        end_hdr = mm.find(b"end_header\n") + len(b"end_header\n")
        dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        total_verts = (len(mm) - end_hdr) // 27
        arr = np.ndarray(buffer=mm, dtype=dt, offset=end_hdr, shape=(total_verts,))
        raw_pts = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float64)
    return raw_pts


# KDL chain joint bounds checking
def build_bounds_checker():
    _, joint_limits, _, _ = build_kdl_chain()

    def satisfies_bounds(q: List[float], tol: float = 1e-5) -> Tuple[bool, int, float, Tuple[float, float]]:
        for j_idx, (lo, hi) in enumerate(joint_limits):
            val = float(q[j_idx])
            if val < (lo - tol) or val > (hi + tol):
                return False, j_idx + 1, val, (lo, hi)
        return True, 0, 0.0, (0.0, 0.0)

    return satisfies_bounds


# Planning scene validity check wrapper
def check_state_validity_multi_dof(
    planner_inst: PiperMoveItPlanner,
    val_client,
    joint_positions: List[float]
) -> Tuple[bool, List[Any]]:
    """
    Query /check_state_validity ensuring req.robot_state.multi_dof_joint_state
    is populated on every call.
    """
    req = GetStateValidity.Request()
    req.group_name = planner_inst.group_name
    req.robot_state.joint_state.name = [f"joint{i+1}" for i in range(6)]
    req.robot_state.joint_state.position = [float(p) for p in joint_positions]
    # Multi-DOF joint state explicitly set on every call site
    req.robot_state.multi_dof_joint_state = planner_inst.get_multi_dof_joint_state()

    fut = val_client.call_async(req)
    rclpy.spin_until_future_complete(planner_inst.node, fut, timeout_sec=5.0)
    res = fut.result()
    if res is not None:
        return res.valid, list(res.contacts)
    return False, []


def serialize_contacts(raw_contacts: List[Any], waypoint_index: int = 0) -> List[Dict[str, Any]]:
    """Serialize MoveIt ContactInformation objects into JSON-compliant dictionary."""
    serialized = []
    for c in raw_contacts:
        pos = [float(c.position.x), float(c.position.y), float(c.position.z)]
        norm = [float(c.normal.x), float(c.normal.y), float(c.normal.z)]
        serialized.append({
            "body_1": str(c.contact_body_1),
            "body_2": str(c.contact_body_2),
            "position_xyz": pos,
            "normal_xyz": norm,
            "depth_m": float(c.depth),
            "frame": "world",
            "waypoint_index": waypoint_index
        })
    return serialized


# Point-in-OBB containment
def point_in_obb(p: np.ndarray, geom: Dict[str, Any]) -> bool:
    c = np.array(geom.get("obb_center") or geom.get("centroid") or geom.get("c"), dtype=float)
    raw_ext = geom.get("extents") or geom.get("d") or (np.array(geom["max"]) - np.array(geom["min"]))
    extents = np.array([max(0.005, float(raw_ext[i])) for i in range(3)], dtype=float)
    yaw = float(geom.get("yaw_rad", 0.0))
    dx, dy, dz = p[0] - c[0], p[1] - c[1], p[2] - c[2]
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    lx = dx * cos_y - dy * sin_y
    ly = dx * sin_y + dy * cos_y
    lz = dz
    hx, hy, hz = extents[0] / 2.0, extents[1] / 2.0, extents[2] / 2.0
    return bool((abs(lx) <= hx) and (abs(ly) <= hy) and (abs(lz) <= hz))


def attribute_contacts(
    contacts: List[Dict[str, Any]],
    rect_node_geoms: Dict[str, Dict[str, Any]],
    belief_assignment: Dict[str, str],
    gt_assignment: Dict[str, str]
) -> Tuple[Optional[str], Optional[str], Optional[str], bool, Optional[bool], str, int, int]:
    """
    Two-branch contact attribution with First-Attributable-Wins semantics:
      - Branch A: If non-robot contact body maps to a published node in rect_node_geoms,
        use that node directly (attribution_method = "named_body"). If named body maps
        to no node in rect_node_geoms (e.g. floor_ground_plane), continues searching.
      - Branch B: If non-robot contact body is <octomap>, fall back to 3D Point-in-OBB containment
        (attribution_method = "point_in_obb").
    Inspects all contacts at the colliding waypoint to record num_external_contacts
    and num_resolved_contacts. The first contact that resolves to a scene node
    determines the trial's attribution and invariant evaluation.
    Returns (attributed_node_id, colliding_node_arm_level, colliding_node_gt_level,
             structural_invariant_verifiable, structural_invariant_passed,
             attribution_method, num_external_contacts, num_resolved_contacts).
    """
    if not contacts:
        return None, None, None, False, None, "unresolved", 0, 0

    # Build mapping from published object ID / suffix to node ID in rect_node_geoms with ambiguity asserts
    obj_id_to_nid = {}
    for nid, geom in rect_node_geoms.items():
        lbl = geom.get("label", "obj")
        key = f"{lbl}_{nid[:8]}"
        if key in obj_id_to_nid:
            raise ValueError(f"Ambiguity collision: prefix key '{key}' already maps to '{obj_id_to_nid[key]}', cannot map to '{nid}'")
        obj_id_to_nid[key] = nid
        if nid in obj_id_to_nid:
            raise ValueError(f"Ambiguity collision: UUID '{nid}' already in obj_id_to_nid")
        obj_id_to_nid[nid] = nid

    candidates = []
    for nid, geom in rect_node_geoms.items():
        raw_ext = geom.get("extents") or geom.get("d") or (np.array(geom["max"]) - np.array(geom["min"]))
        vol = float(raw_ext[0] * raw_ext[1] * raw_ext[2])
        candidates.append((vol, nid, geom))
    candidates.sort(key=lambda x: x[0])

    num_external_contacts = 0
    num_resolved_contacts = 0
    first_resolved = None

    for c in contacts:
        b1 = c.get("body_1", "")
        b2 = c.get("body_2", "")
        if b1 in ALL_ROBOT_LINKS and b2 not in ALL_ROBOT_LINKS:
            ext_body = b2
        elif b2 in ALL_ROBOT_LINKS and b1 not in ALL_ROBOT_LINKS:
            ext_body = b1
        elif b1 not in ALL_ROBOT_LINKS:
            ext_body = b1
        elif b2 not in ALL_ROBOT_LINKS:
            ext_body = b2
        else:
            ext_body = ""

        if not ext_body:
            continue

        num_external_contacts += 1

        # BRANCH A: Named CollisionObject
        if ext_body != "<octomap>":
            nid = obj_id_to_nid.get(ext_body)
            if nid is None:
                for cand_nid in rect_node_geoms:
                    if ext_body.endswith(f"_{cand_nid[:8]}"):
                        if nid is not None and nid != cand_nid:
                            raise ValueError(f"Ambiguity collision in suffix match: ext_body '{ext_body}' matches '{nid}' and '{cand_nid}'")
                        nid = cand_nid
            if nid is not None:
                num_resolved_contacts += 1
                if first_resolved is None:
                    arm_lvl = belief_assignment.get(nid)
                    gt_lvl = gt_assignment.get(nid)
                    if arm_lvl is None:
                        first_resolved = (nid, None, gt_lvl, False, None, "named_body")
                    else:
                        inv_passed = (arm_lvl not in ("point_cloud", "bounding_box"))
                        first_resolved = (nid, arm_lvl, gt_lvl, True, inv_passed, "named_body")
            else:
                # Unmapped named body (e.g. floor_ground_plane) — continue searching remaining contacts
                continue

        # BRANCH B: <octomap> fallback via 3D Point-in-OBB
        elif ext_body == "<octomap>":
            p = np.array(c["position_xyz"], dtype=float)
            octo_nid = None
            for _, cand_nid, geom in candidates:
                if point_in_obb(p, geom):
                    octo_nid = cand_nid
                    break

            if octo_nid is not None:
                num_resolved_contacts += 1
                if first_resolved is None:
                    arm_lvl = belief_assignment.get(octo_nid)
                    gt_lvl = gt_assignment.get(octo_nid)
                    if arm_lvl is None:
                        first_resolved = (octo_nid, None, gt_lvl, False, None, "point_in_obb")
                    else:
                        inv_passed = (arm_lvl not in ("point_cloud", "bounding_box"))
                        first_resolved = (octo_nid, arm_lvl, gt_lvl, True, inv_passed, "point_in_obb")
            else:
                continue

    if first_resolved is not None:
        nid, arm_lvl, gt_lvl, verifiable, passed, method = first_resolved
        return nid, arm_lvl, gt_lvl, verifiable, passed, method, num_external_contacts, num_resolved_contacts
    else:
        # Unattributed / unresolved contact
        return None, None, None, False, None, "unresolved", num_external_contacts, 0


# Truth Scene Caching
class TruthSceneCache:
    def __init__(self, scene_builder: MoveItSceneBuilder):
        self.scene_builder = scene_builder
        self.cached_key: Optional[Tuple[str, str, str]] = None
        self.cached_msg: Optional[PlanningScene] = None
        self.cached_meta: Optional[Dict[str, Any]] = None

    def get_truth_scene(
        self,
        scene_id: str,
        target_aff_id: str,
        target_obj_id: str,
        scene_nodes: List[dict],
        rect_pts: np.ndarray,
        rect_node_geoms: Dict[str, dict]
    ) -> Tuple[PlanningScene, Dict[str, Any]]:
        current_key = (scene_id, target_aff_id, target_obj_id)
        if self.cached_key == current_key:
            # Assert key match on every cache hit
            assert self.cached_key == (scene_id, target_aff_id, target_obj_id), "Truth scene cache key mismatch!"
            return self.cached_msg, self.cached_meta

        msg, meta = self.scene_builder.build_truth_scene_data(
            scene_id=scene_id,
            scene_nodes=scene_nodes,
            rect_pts=rect_pts,
            node_geometries=rect_node_geoms,
            target_aff_id=target_aff_id,
            target_obj_id=target_obj_id,
            use_acm=True,
            octree_res=0.01
        )
        self.cached_key = current_key
        self.cached_msg = msg
        self.cached_meta = meta
        return self.cached_msg, self.cached_meta


# 10-step Bisection obstacle clearance computation
def bisect_rung6_clearance(
    planner_inst: PiperMoveItPlanner,
    apply_scene_client,
    val_client,
    waypoints: List[List[float]],
    robot_links: List[str],
    max_padding: float = 0.20,
    steps: int = 10
) -> float:
    def set_padding(p: float) -> bool:
        scene_diff = PlanningScene()
        scene_diff.is_diff = True
        for l in robot_links:
            lp = LinkPadding()
            lp.link_name = l
            lp.padding = float(p)
            scene_diff.link_padding.append(lp)
        req = ApplyPlanningScene.Request()
        req.scene = scene_diff
        fut = apply_scene_client.call_async(req)
        rclpy.spin_until_future_complete(planner_inst.node, fut, timeout_sec=5.0)
        return fut.result().success if fut.result() else False

    low = 0.0
    high = max_padding
    for _ in range(1, steps + 1):
        mid = (low + high) / 2.0
        set_padding(mid)
        all_valid = True
        for wp in waypoints:
            valid, _ = check_state_validity_multi_dof(planner_inst, val_client, wp)
            if not valid:
                all_valid = False
                break
        if all_valid:
            low = mid
        else:
            high = mid

    # Reset padding to 0.0 after bisection
    set_padding(0.0)
    return low


def main():
    parser = argparse.ArgumentParser(description="Planning Benchmark Batch Runner")
    parser.add_argument("--cases-file", type=str, default=None, help="Path to file listing case IDs (one per line)")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated list of case IDs")
    parser.add_argument("--limit-cases", type=int, default=None, help="Limit to first N cases")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per case-arm pair (default: 3)")
    parser.add_argument("--output-file", type=str, default="planning_benchmark_results.jsonl", help="Output JSONL path")
    parser.add_argument("--restart-interval", type=int, default=8, help="Planned restart interval in trials (default: 8)")

    args = parser.parse_args()

    print("================================================================================", flush=True)
    print("=== PLANNING BENCHMARK BATCH RUNNER (run_planning_benchmark.py) ===", flush=True)
    print("================================================================================", flush=True)

    # 1. Ingestion Hash Assertions
    print("\n--- 1. INGESTION HASH VERIFICATION ---", flush=True)
    kin_hash = compute_file_sha256(KINEMATICS_PATH)
    assert kin_hash == EXPECTED_KINEMATICS_HASH, (
        f"Kinematics hash mismatch! Expected {EXPECTED_KINEMATICS_HASH}, got {kin_hash}"
    )
    print(f"  Kinematics Hash: {kin_hash} (PASSED)", flush=True)

    defaults_hash = compute_file_sha256(DEFAULTS_PATH)
    assert defaults_hash == EXPECTED_DEFAULTS_HASH, (
        f"Defaults hash mismatch! Expected {EXPECTED_DEFAULTS_HASH}, got {defaults_hash}"
    )
    print(f"  Defaults Hash:   {defaults_hash} (PASSED)", flush=True)

    # Compute runner self hash
    runner_hash = compute_file_sha256(Path(__file__))
    print(f"  Runner Self Hash:{runner_hash}", flush=True)

    # Git commit SHA
    try:
        git_commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_commit_sha = "unknown"
    print(f"  Git Commit SHA:  {git_commit_sha}", flush=True)

    # Load artifacts
    with open(KINEMATICS_PATH) as f:
        reselected_kin = json.load(f)["cases"]
    with open(DEFAULTS_PATH) as f:
        defaults_eval = json.load(f)

    defaults_map = {item.get("filename") or item.get("case_id"): item for item in defaults_eval}

    # 2. Case Resolution & Positional Slice Guard
    if args.cases_file:
        with open(args.cases_file) as f:
            case_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    elif args.cases:
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_ids = list(reselected_kin.keys())

    if args.limit_cases is not None:
        case_ids = case_ids[:args.limit_cases]
        distinct_scenes = set(reselected_kin[c]["scene"] for c in case_ids if c in reselected_kin)
        if len(case_ids) >= 5 and len(distinct_scenes) < 5:
            raise ValueError(
                f"Positional slice guard tripped: {len(case_ids)} cases cover only {len(distinct_scenes)} "
                f"distinct scenes (< 5). Explicit case list required."
            )

    print(f"\n--- 2. RESOLVED CASE LIST ({len(case_ids)} cases) ---", flush=True)
    for idx, cid in enumerate(case_ids):
        sc = reselected_kin[cid]["scene"]
        print(f"  [{idx+1:2d}/{len(case_ids)}] {cid} (scene: {sc})", flush=True)

    # 3. Policy Arms & Ground Truth Assignment Ingestion Verification
    print("\n--- 3. POLICY ARMS VERIFICATION ---", flush=True)
    for cid in case_ids:
        c_eval = defaults_map.get(cid)
        if not c_eval:
            raise ValueError(f"Case '{cid}' not found in defaults artifact!")
        if "gt_assignment" not in c_eval or not isinstance(c_eval["gt_assignment"], dict):
            raise ValueError(f"Required key 'gt_assignment' missing or invalid in defaults artifact for case '{cid}'!")
        for arm in PINNED_EVAL_ARMS:
            if arm not in c_eval.get("policy_outputs", {}):
                raise ValueError(f"Arm '{arm}' not found in policy_outputs for case '{cid}'!")
    print(f"  All {len(PINNED_EVAL_ARMS)} pinned arms verified in defaults artifact: {PINNED_EVAL_ARMS}", flush=True)

    # 4. Initialize Process & ROS 2 (Unconditional clean initial restart)
    print("\n--- 4. INITIAL PROCESS SETUP ---", flush=True)
    current_pid, _ = restart_move_group_process()

    if not rclpy.ok():
        rclpy.init()

    node = Node("planning_benchmark_runner")
    planner = PiperMoveItPlanner(node, group_name="arm", world_frame="world")
    planner.wait_for_services(timeout_sec=10.0)

    scene_builder = MoveItSceneBuilder(node)
    truth_cache = TruthSceneCache(scene_builder)

    apply_scene_client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    apply_scene_client.wait_for_service(timeout_sec=10.0)

    val_client = node.create_client(GetStateValidity, "/check_state_validity")
    val_client.wait_for_service(timeout_sec=10.0)

    get_scene_client = node.create_client(GetPlanningScene, "/get_planning_scene")
    get_scene_client.wait_for_service(timeout_sec=10.0)

    satisfies_bounds = build_bounds_checker()

    # Preload and Rectify Scene Geometries
    print("\n--- 5. PRELOADING SCENE GEOMETRIES ---", flush=True)
    unique_scenes = sorted(list(set(reselected_kin[c]["scene"] for c in case_ids)))
    scene_cache = {}
    for sc in unique_scenes:
        # Single z_floor source from affordance_heights.json
        z_floor = load_floor_elevation(str(HEIGHTS_PATH), sc)
        # Raw PLY point cloud loading
        raw_pts = load_ply_points(str(DATASET_PATH), sc)
        # Rectify points and node geoms by single z_floor
        rect_pts = raw_pts.copy()
        rect_pts[:, 2] -= z_floor
        annots = load_dataset_annotations(str(DATASET_PATH), sc)
        node_geoms = load_node_geometries(str(GEOM_PATH), sc)
        rect_node_geoms = rectify_node_geometries(node_geoms, z_floor=z_floor)
        scene_cache[sc] = {
            "z_floor": z_floor,
            "rect_pts": rect_pts,
            "annots": annots,
            "rect_node_geoms": rect_node_geoms
        }
        print(f"  Scene: {sc:<14} | z_floor = {z_floor:+.6f} m | Points: {len(rect_pts):7d} | Nodes: {len(rect_node_geoms):2d}", flush=True)

    # Prepare Output JSONL
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(out_path, "w", buffering=1)

    total_trials = len(case_ids) * len(PINNED_EVAL_ARMS) * args.repeats
    print(f"\n--- 6. EXECUTING BATCH ({total_trials} total trials) ---", flush=True)

    trial_counter = 0
    restart_index = 0
    batch_t0 = time.time()

    for cid in case_ids:
        c_kin = reselected_kin[cid]
        c_eval = defaults_map[cid]
        sc = c_kin["scene"]
        task_name = c_kin.get("task", c_eval.get("task", "unknown"))
        aff_id = c_kin["target_aff_id"]
        target_obj_id = c_eval.get("target_obj_id", aff_id)

        # Base placement integrity
        bp = c_kin["base_pose"]
        # Base elevation h interpreted as vertical distance above floor (world Z = z_floor + h)
        bx, by, h, yaw = float(bp["base_x"]), float(bp["base_y"]), float(bp["h"]), float(bp["base_yaw_deg"])
        # Standoff to tool center point pre-measured to link6 flange origin
        q_target = [float(x) for x in c_kin["reselected_target_joint_configuration"]]
        pitch_deg = float(c_kin.get("reselected_pitch_deg", 0.0))
        roll_deg = float(c_kin.get("reselected_roll_deg", 0.0))

        sc_data = scene_cache[sc]
        z_floor = sc_data["z_floor"]
        rect_pts = sc_data["rect_pts"]
        annots = sc_data["annots"]
        rect_node_geoms = sc_data["rect_node_geoms"]

        for arm in PINNED_EVAL_ARMS:
            for rep in range(1, args.repeats + 1):
                # Check if planned restart is due between trials
                if trial_counter > 0 and (trial_counter % args.restart_interval == 0):
                    print(f"\n[PLANNED RESTART] Trial count {trial_counter}: Triggering planned restart {restart_index + 1}...", flush=True)
                    current_pid, _ = restart_move_group_process()
                    restart_index += 1
                    # Reconnect planner and service clients
                    planner.wait_for_services(timeout_sec=10.0)
                    apply_scene_client.wait_for_service(timeout_sec=10.0)
                    val_client.wait_for_service(timeout_sec=10.0)
                    get_scene_client.wait_for_service(timeout_sec=10.0)

                # Unplanned process termination assertion
                live_pid = get_move_group_pid()
                if live_pid != current_pid:
                    raise RuntimeError(
                        f"CRITICAL ERROR: Unplanned move_group PID change detected! "
                        f"Expected {current_pid}, got {live_pid}. Immediate process halt."
                    )

                trial_t0 = time.time()
                rss_val = get_rss_mb(current_pid)

                # Robot base frame orientation and pose configuration
                planner.set_robot_base_pose(bx, by, h, yaw)
                tf = planner.get_multi_dof_joint_state().transforms[0]
                assert abs(tf.translation.x - bx) < 1e-4, f"Base X mismatch: {tf.translation.x} vs {bx}"
                assert abs(tf.translation.y - by) < 1e-4, f"Base Y mismatch: {tf.translation.y} vs {by}"
                assert abs(tf.translation.z - h) < 1e-4, f"Base Z mismatch: {tf.translation.z} vs {h}"

                # Build belief scene
                detail_assignment = c_eval["policy_outputs"][arm]["assignment"]
                # Non-empty detail assignment assertion
                assert len(detail_assignment) > 0, f"Detail assignment is empty for scene {sc}, arm {arm}"
                all_remove_scene = all(lvl in ("remove", "label") for lvl in detail_assignment.values())

                # Build planning scene with SRDF self-collision preservation and binary octree serialization
                belief_msg, belief_meta = scene_builder.build_planning_scene_data(
                    scene_id=sc,
                    scene_nodes=annots,
                    rect_pts=rect_pts,
                    node_geometries=rect_node_geoms,
                    detail_assignment=detail_assignment,
                    target_aff_id=aff_id,
                    target_obj_id=target_obj_id,
                    use_acm=True
                )
                built_leaves = belief_meta.get("octree_leaf_count", 0)
                scene_builder.load_scene(belief_msg)

                # Resident octomap leaf count assertion (with query retry & extended timeout)
                req_ps = GetPlanningScene.Request()
                req_ps.components.components = PlanningSceneComponents.OCTOMAP
                res_ps = None
                max_query_attempts = 3
                for attempt in range(1, max_query_attempts + 1):
                    if not get_scene_client.service_is_ready():
                        get_scene_client.wait_for_service(timeout_sec=5.0)
                    fut_ps = get_scene_client.call_async(req_ps)
                    rclpy.spin_until_future_complete(node, fut_ps, timeout_sec=15.0)
                    res_ps = fut_ps.result()
                    if res_ps is not None:
                        break
                    print(f"WARNING: /get_planning_scene query timed out (attempt {attempt}/{max_query_attempts}), retrying...", flush=True)
                    time.sleep(1.0)
                assert res_ps is not None, "Failed to query /get_planning_scene for octomap leaf count check"
                resident_leaves = get_resident_octree_leaf_count(res_ps.scene.world.octomap.octomap)
                if resident_leaves != built_leaves:
                    raise AssertionError(
                        f"Octree leaf count mismatch: Built leaves {built_leaves} != Resident leaves {resident_leaves} "
                        f"(Scene: {sc}, Arm: {arm})"
                    )

                # Plan in belief scene
                t0_plan = time.time()
                plan_ok, plan_metrics, _ = planner.plan_to_joint_goal(
                    q_target,
                    planner_id="RRTConnect",
                    allowed_planning_time=3.0,
                    num_planning_attempts=5
                )
                planning_time_wall_sec = time.time() - t0_plan

                # Initialize outcome fields
                outcome = "UNKNOWN"
                wps = []
                num_waypoints = 0
                path_length_rad = 0.0
                all_bounds_ok = None
                contacts = []
                num_external_contacts = 0
                num_resolved_contacts = 0
                attributed_node_id = None
                attribution_method = "n/a"
                colliding_node_arm_level = None
                colliding_node_gt_level = None
                structural_invariant_verifiable = None
                structural_invariant_passed = None
                r6_clearance_m = None
                goal_valid_in_belief = None
                goal_contacts = []

                if not plan_ok or not plan_metrics:
                    outcome = "BELIEF_PLAN_FAILED"
                    # Direct goal state validity measurement
                    goal_valid, goal_raw_c = check_state_validity_multi_dof(planner, val_client, q_target)
                    goal_valid_in_belief = goal_valid
                    goal_contacts = serialize_contacts(goal_raw_c)
                else:
                    wps = plan_metrics["waypoints"]
                    num_waypoints = len(wps)
                    # Compute path length
                    path_len = 0.0
                    for i in range(len(wps) - 1):
                        path_len += float(np.linalg.norm(np.array(wps[i+1]) - np.array(wps[i])))
                    path_length_rad = path_len

                    # Check bounds for every waypoint
                    all_bounds_ok = all(satisfies_bounds(wp, tol=1e-5)[0] for wp in wps)

                    # Swap to Ground Truth World
                    truth_msg, truth_meta = truth_cache.get_truth_scene(
                        scene_id=sc,
                        target_aff_id=aff_id,
                        target_obj_id=target_obj_id,
                        scene_nodes=annots,
                        rect_pts=rect_pts,
                        rect_node_geoms=rect_node_geoms
                    )
                    scene_builder.load_scene(truth_msg)

                    # Assert base pose preserved after truth-scene load before validity checking
                    tf2 = planner.get_multi_dof_joint_state().transforms[0]
                    assert abs(tf2.translation.x - bx) < 1e-4, f"Truth scene Base X mismatch: {tf2.translation.x} vs {bx}"
                    assert abs(tf2.translation.y - by) < 1e-4, f"Truth scene Base Y mismatch: {tf2.translation.y} vs {by}"
                    assert abs(tf2.translation.z - h) < 1e-4, f"Truth scene Base Z mismatch: {tf2.translation.z} vs {h}"

                    # Validate trajectory in truth scene
                    truth_valid = True
                    colliding_wp_idx = -1
                    colliding_raw_contacts = []
                    for wp_idx, wp in enumerate(wps):
                        valid, raw_c = check_state_validity_multi_dof(planner, val_client, wp)
                        if not valid:
                            truth_valid = False
                            colliding_wp_idx = wp_idx
                            colliding_raw_contacts = raw_c
                            break

                    if not truth_valid:
                        outcome = "UNSAFE_COLLISION"
                        contacts = serialize_contacts(colliding_raw_contacts, waypoint_index=colliding_wp_idx)
                        # Contact attribution via Point-in-OBB
                        assert "gt_assignment" in c_eval, f"Missing gt_assignment in case {cid}"
                        (attributed_node_id,
                         colliding_node_arm_level,
                         colliding_node_gt_level,
                         structural_invariant_verifiable,
                         structural_invariant_passed,
                         attribution_method,
                         num_external_contacts,
                         num_resolved_contacts) = attribute_contacts(
                            contacts, rect_node_geoms, detail_assignment, c_eval["gt_assignment"]
                        )
                    elif not all_bounds_ok:
                        outcome = "OUT_OF_BOUNDS_COLLISION_FREE"
                    else:
                        outcome = "SAFE"
                        r6_clearance_m = bisect_rung6_clearance(
                            planner, apply_scene_client, val_client, wps, ROBOT_LINKS, max_padding=0.20, steps=10
                        )

                # On any path where no contact is attributed, set verifiable = False and passed = None
                if attributed_node_id is None:
                    structural_invariant_verifiable = False
                    structural_invariant_passed = None

                trial_wall_clock_sec = time.time() - trial_t0

                # Provenance block
                provenance = {
                    "build_hash_scene_builder": BUILD_HASH_SCENE_BUILDER,
                    "build_hash_batch_runner": runner_hash,
                    "build_hash_octomap_so": BUILD_HASH_OCTOMAP_SO,
                    "build_hash_piper_srdf": BUILD_HASH_PIPER_SRDF,
                    "defaults_hash": EXPECTED_DEFAULTS_HASH,
                    "kinematics_hash": EXPECTED_KINEMATICS_HASH,
                    "git_commit_sha": git_commit_sha,
                    "wall_clock_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }

                # Construct JSONL Record
                record = {
                    "case_id": cid,
                    "scene": sc,
                    "task": task_name,
                    "target_aff_id": aff_id,
                    "target_obj_id": target_obj_id,
                    "arm": arm,
                    "repeat_index": rep,
                    "base_pose": {"base_x": bx, "base_y": by, "h": h, "base_yaw_deg": yaw},
                    "base_pose_source": "ground_truth_kinematics_reselected.json",
                    "base_pose_asserted": True,
                    "selected_pitch_deg": pitch_deg,
                    "selected_roll_deg": roll_deg,
                    "joint_target": q_target,
                    "belief_octree_leaf_count": built_leaves,
                    "resident_octree_leaf_count": resident_leaves,
                    "all_remove_scene": all_remove_scene,
                    "belief_plan_success": bool(plan_ok),
                    "outcome": outcome,
                    "num_waypoints": num_waypoints,
                    "path_length_rad": round(path_length_rad, 4),
                    "planning_time_wall_sec": round(planning_time_wall_sec, 4),
                    "trial_wall_clock_sec": round(trial_wall_clock_sec, 4),
                    "all_waypoints_satisfy_bounds": all_bounds_ok,
                    "waypoints": wps,
                    "contacts": contacts,
                    "num_external_contacts": num_external_contacts,
                    "num_resolved_contacts": num_resolved_contacts,
                    "attributed_node_id": attributed_node_id,
                    "attribution_method": attribution_method,
                    "colliding_node_arm_level": colliding_node_arm_level,
                    "colliding_node_gt_level": colliding_node_gt_level,
                    "structural_invariant_verifiable": structural_invariant_verifiable,
                    "structural_invariant_passed": structural_invariant_passed,
                    "r6_clearance_m": round(r6_clearance_m, 6) if r6_clearance_m is not None else None,
                    "goal_valid_in_belief": goal_valid_in_belief,
                    "goal_contacts": goal_contacts,
                    "move_group_pid": current_pid,
                    "restart_index": restart_index,
                    "rss_mb": round(rss_val, 2),
                    "provenance": provenance
                }

                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

                trial_counter += 1
                clr_str = f"{r6_clearance_m*1000:.3f} mm" if r6_clearance_m is not None else "N/A"
                print(
                    f"[{trial_counter:2d}/{total_trials}] {cid:<35} | {arm:<26} | "
                    f"Outcome: {outcome:<18} | Leaves: {resident_leaves:6d} | Clr: {clr_str:<9} | "
                    f"t_trial: {trial_wall_clock_sec:.3f} s",
                    flush=True
                )

    out_file.close()
    batch_wall_sec = time.time() - batch_t0
    print("\n================================================================================", flush=True)
    print(f"=== BATCH COMPLETE: {trial_counter} trials executed in {batch_wall_sec:.2f} s ({batch_wall_sec/60:.2f} min) ===", flush=True)
    print(f"=== Output written to: {out_path.resolve()} ===", flush=True)
    print("================================================================================", flush=True)


if __name__ == "__main__":
    main()
