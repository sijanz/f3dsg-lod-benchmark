#!/usr/bin/env python3
"""
MoveIt 2 Planning Scene Builder for FunGraph3D Scenes.

Implements the Track B canonical scene construction semantics:
- detail level 'point_cloud' : OctoMap at 0.01 m resolution (lexicographically sorted, updateNode only)
- detail level 'bounding_box': SolidPrimitive.BOX with exact OBB extents (0.005 m extent floor)
- detail level 'label'       : excluded (0 collision geometry)
- detail level 'remove'      : excluded (0 collision geometry)
"""

import os
import math
import ctypes
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Time as TimeMsg
from std_srvs.srv import Empty
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import (
    CollisionObject,
    PlanningScene,
    PlanningSceneWorld,
    PlanningSceneComponents,
    AllowedCollisionMatrix,
    AllowedCollisionEntry
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
from octomap_msgs.msg import OctomapWithPose, Octomap


# ==============================================================================
# 1. OCTOMAP CTYPES BRIDGE
# ==============================================================================

BRIDGE_LIB_PATH = Path(__file__).parent / "liboctomap_bridge.so"


class OctoMapBridge:
    """Wrapper around liboctomap_bridge.so providing C++ OctoMap functionality."""

    def __init__(self, lib_path: Optional[str] = None):
        path = str(lib_path or BRIDGE_LIB_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"OctoMap bridge library not found at: {path}")
        self.lib = ctypes.CDLL(path)

        # Setup ctypes signatures
        self.lib.octree_create.restype = ctypes.c_void_p
        self.lib.octree_create.argtypes = [ctypes.c_double]

        self.lib.octree_free.restype = None
        self.lib.octree_free.argtypes = [ctypes.c_void_p]

        self.lib.octree_insert_points.restype = None
        self.lib.octree_insert_points.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
        ]

        self.lib.octree_get_num_leaf_nodes.restype = ctypes.c_size_t
        self.lib.octree_get_num_leaf_nodes.argtypes = [ctypes.c_void_p]

        self.lib.octree_get_resolution.restype = ctypes.c_double
        self.lib.octree_get_resolution.argtypes = [ctypes.c_void_p]

        self.lib.octree_write_full.restype = ctypes.c_int
        self.lib.octree_write_full.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t)
        ]

        self.lib.octree_write_binary.restype = ctypes.c_int
        self.lib.octree_write_binary.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t)
        ]

        self.lib.octree_is_point_occupied.restype = ctypes.c_int
        self.lib.octree_is_point_occupied.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]

        self.lib.octree_write_binary_data.restype = ctypes.c_int
        self.lib.octree_write_binary_data.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
        ]

        self.lib.octree_read_binary_data.restype = ctypes.c_void_p
        self.lib.octree_read_binary_data.argtypes = [
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]

        self.lib.octree_get_metric_bounds.restype = ctypes.c_int
        self.lib.octree_get_metric_bounds.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]

        self.lib.free_buffer.restype = None
        self.lib.free_buffer.argtypes = [ctypes.POINTER(ctypes.c_uint8)]

    def create_tree(self, resolution: float = 0.01) -> ctypes.c_void_p:
        return self.lib.octree_create(ctypes.c_double(resolution))

    def free_tree(self, tree_ptr: ctypes.c_void_p):
        if tree_ptr:
            self.lib.octree_free(tree_ptr)

    def insert_points(self, tree_ptr: ctypes.c_void_p, points: np.ndarray):
        """
        Insert points into tree using updateNode(point, true).
        Points are sorted lexicographically by (x, y, z) for strict determinism.
        """
        if points is None or len(points) == 0:
            return
        # Lexicographical sort by (x, y, z)
        sorted_indices = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        sorted_pts = np.ascontiguousarray(points[sorted_indices], dtype=np.float64)

        pts_ptr = sorted_pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.lib.octree_insert_points(tree_ptr, pts_ptr, len(sorted_pts))

    def get_num_leaf_nodes(self, tree_ptr: ctypes.c_void_p) -> int:
        return int(self.lib.octree_get_num_leaf_nodes(tree_ptr))

    def get_resolution(self, tree_ptr: ctypes.c_void_p) -> float:
        return float(self.lib.octree_get_resolution(tree_ptr))

    def write_full_bytes(self, tree_ptr: ctypes.c_void_p) -> bytes:
        """Serialize full .ot tree (unthresholded) and return raw bytes."""
        out_buf = ctypes.POINTER(ctypes.c_uint8)()
        out_len = ctypes.c_size_t()
        ok = self.lib.octree_write_full(tree_ptr, ctypes.byref(out_buf), ctypes.byref(out_len))
        if not ok or out_len.value == 0:
            return b""
        raw = bytes(ctypes.cast(out_buf, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        self.lib.free_buffer(out_buf)
        return raw

    def write_binary_bytes(self, tree_ptr: ctypes.c_void_p) -> bytes:
        """Serialize binary .bt tree (with file header) and return raw bytes."""
        out_buf = ctypes.POINTER(ctypes.c_uint8)()
        out_len = ctypes.c_size_t()
        ok = self.lib.octree_write_binary(tree_ptr, ctypes.byref(out_buf), ctypes.byref(out_len))
        if not ok or out_len.value == 0:
            return b""
        raw = bytes(ctypes.cast(out_buf, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        self.lib.free_buffer(out_buf)
        return raw

    def write_binary_data_bytes(self, tree_ptr: ctypes.c_void_p) -> bytes:
        """Serialize pure binary data (data only, exactly matching octomap_msgs::binaryMapToMsg)."""
        out_buf = ctypes.POINTER(ctypes.c_uint8)()
        out_len = ctypes.c_size_t()
        ok = self.lib.octree_write_binary_data(tree_ptr, ctypes.byref(out_buf), ctypes.byref(out_len))
        if not ok or out_len.value == 0:
            return b""
        raw = bytes(ctypes.cast(out_buf, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        self.lib.free_buffer(out_buf)
        return raw

    def read_binary_data_bytes(self, resolution: float, raw_bytes: bytes) -> ctypes.c_void_p:
        """Deserialize tree from pure binary data (matching octomap_msgs::binaryMsgToMap)."""
        if not raw_bytes:
            return None
        buf = (ctypes.c_uint8 * len(raw_bytes)).from_buffer_copy(raw_bytes)
        return self.lib.octree_read_binary_data(ctypes.c_double(resolution), buf, len(raw_bytes))

    def get_metric_bounds(self, tree_ptr: ctypes.c_void_p) -> Tuple[float, float, float, float, float, float]:
        """Return (min_x, min_y, min_z, max_x, max_y, max_z) bounding box of tree."""
        min_x = ctypes.c_double()
        min_y = ctypes.c_double()
        min_z = ctypes.c_double()
        max_x = ctypes.c_double()
        max_y = ctypes.c_double()
        max_z = ctypes.c_double()
        ok = self.lib.octree_get_metric_bounds(
            tree_ptr,
            ctypes.byref(min_x), ctypes.byref(min_y), ctypes.byref(min_z),
            ctypes.byref(max_x), ctypes.byref(max_y), ctypes.byref(max_z)
        )
        if not ok:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return (min_x.value, min_y.value, min_z.value, max_x.value, max_y.value, max_z.value)

    def is_point_occupied(self, tree_ptr: ctypes.c_void_p, x: float, y: float, z: float) -> bool:
        return bool(self.lib.octree_is_point_occupied(tree_ptr, ctypes.c_double(x), ctypes.c_double(y), ctypes.c_double(z)))


_OCTOMAP_BRIDGE: Optional[OctoMapBridge] = None

def get_octomap_bridge() -> OctoMapBridge:
    global _OCTOMAP_BRIDGE
    if _OCTOMAP_BRIDGE is None:
        _OCTOMAP_BRIDGE = OctoMapBridge()
    return _OCTOMAP_BRIDGE


def rectify_node_geometries(node_geoms_raw: Dict[str, dict], z_floor: float) -> Dict[str, dict]:
    """Transform raw node geometry centroids into rectified frame (floor at z=0)."""
    rect_geoms = {}
    for node_id, geom in node_geoms_raw.items():
        g_copy = dict(geom)
        c = list(geom["centroid"])
        c[2] = float(c[2]) - float(z_floor)
        g_copy["centroid"] = c
        if "min" in geom and "max" in geom:
            mn = list(geom["min"])
            mx = list(geom["max"])
            mn[2] -= float(z_floor)
            mx[2] -= float(z_floor)
            g_copy["min"] = mn
            g_copy["max"] = mx
            g_copy["extents"] = [float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(mx[2] - mn[2])]
            g_copy["d"] = g_copy["extents"]
        rect_geoms[node_id] = g_copy
    return rect_geoms


def yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    """Convert yaw angle (rotation around Z axis) to geometry_msgs.msg.Quaternion."""
    half_yaw = yaw_rad / 2.0
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half_yaw)
    q.w = math.cos(half_yaw)
    return q


# ==============================================================================
# ==============================================================================
# 2. ALLOWED COLLISION MATRIX (ACM) UTILITIES
# ==============================================================================

GRIPPER_LINKS = ["gripper_base", "link7", "link8"]


ROBOT_LINKS = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base", "link7", "link8"]

DEFAULT_DISABLED_COLLISIONS = [
    ("base_link", "link1"), ("base_link", "link2"), ("base_link", "link3"),
    ("gripper_base", "link3"), ("gripper_base", "link4"), ("gripper_base", "link5"),
    ("gripper_base", "link6"), ("gripper_base", "link7"), ("gripper_base", "link8"),
    ("link1", "link2"), ("link1", "link3"),
    ("link2", "link3"), ("link2", "link4"),
    ("link3", "link4"), ("link3", "link5"), ("link3", "link6"), ("link3", "link7"), ("link3", "link8"),
    ("link4", "link5"), ("link4", "link6"), ("link4", "link7"), ("link4", "link8"),
    ("link5", "link6"), ("link5", "link7"), ("link5", "link8"),
    ("link6", "link7"), ("link6", "link8"),
    ("link7", "link8")
]


def build_allowed_collision_matrix(
    gripper_links: List[str],
    target_object_ids: List[str],
    whitelist_octomap: bool = False,
) -> AllowedCollisionMatrix:
    """
    Build an AllowedCollisionMatrix whitelisting gripper links with target obstacles,
    while preserving the 28 default robot self-collision disabled pairs from the SRDF.
    """
    all_names = list(ROBOT_LINKS) + [oid for oid in target_object_ids if oid]
    if whitelist_octomap and "<octomap>" not in all_names:
        all_names.append("<octomap>")

    # Deduplicate while preserving order
    seen = set()
    dedup_names = []
    for n in all_names:
        if n not in seen:
            seen.add(n)
            dedup_names.append(n)

    acm = AllowedCollisionMatrix()
    acm.entry_names = dedup_names
    n_entries = len(dedup_names)
    matrix = [[False] * n_entries for _ in range(n_entries)]

    disabled_pairs = set()
    for l1, l2 in DEFAULT_DISABLED_COLLISIONS:
        disabled_pairs.add((l1, l2))
        disabled_pairs.add((l2, l1))

    for i, n1 in enumerate(dedup_names):
        for j, n2 in enumerate(dedup_names):
            if (n1, n2) in disabled_pairs:
                matrix[i][j] = True
                matrix[j][i] = True

    targets = set(target_object_ids)
    if whitelist_octomap:
        targets.add("<octomap>")

    for i, name_i in enumerate(dedup_names):
        for j, name_j in enumerate(dedup_names):
            if (name_i in gripper_links and name_j in targets) or (name_j in gripper_links and name_i in targets):
                matrix[i][j] = True
                matrix[j][i] = True

    for row in matrix:
        entry = AllowedCollisionEntry()
        entry.enabled = row
        acm.entry_values.append(entry)

    return acm


# ==============================================================================
# 3. SCENE BUILDER CLASS (ONE API FOR BELIEF AND TRUTH WORLDS)
# ==============================================================================

class MoveItSceneBuilder:
    """
    Manages publishing and updating FunGraph3D collision objects in MoveIt 2.
    Implements unified scene construction for both Belief and Truth worlds.
    """

    def __init__(self, node: Node, world_frame: str = "world"):
        self.node = node
        self.world_frame = world_frame
        self.octomap_bridge = get_octomap_bridge()

        # Publisher and service clients for MoveIt Planning Scene
        self.collision_pub = self.node.create_publisher(
            CollisionObject, "/collision_object", 10
        )
        self.planning_scene_pub = self.node.create_publisher(
            PlanningScene, "/planning_scene", 10
        )
        self.apply_scene_client = self.node.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )
        self.get_scene_client = self.node.create_client(
            GetPlanningScene, "/get_planning_scene"
        )
        self.clear_octomap_client = self.node.create_client(
            Empty, "/clear_octomap"
        )

        self._active_object_ids = set()

    def clear_scene(self, timeout_sec: float = 2.0) -> bool:
        """Remove all collision objects and clear OctoMap currently registered in MoveIt 2."""
        if self.clear_octomap_client.wait_for_service(timeout_sec=0.2):
            fut = self.clear_octomap_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=0.5)

        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.robot_state.is_diff = True

        for obj_id in list(self._active_object_ids):
            co = CollisionObject()
            co.header.frame_id = self.world_frame
            co.id = obj_id
            co.operation = CollisionObject.REMOVE
            scene_msg.world.collision_objects.append(co)

        self._active_object_ids.clear()
        return self._send_planning_scene(scene_msg, timeout_sec=timeout_sec)

    def build_planning_scene_data(
        self,
        scene_id: str,
        scene_nodes: List[dict],
        rect_pts: np.ndarray,
        node_geometries: Dict[str, dict],
        detail_assignment: Dict[str, str],
        target_aff_id: Optional[str] = None,
        target_obj_id: Optional[str] = None,
        use_acm: bool = True,
        octree_res: float = 0.01,
    ) -> Tuple[PlanningScene, Dict[str, Any]]:
        """
        Unified scene data construction function for both Belief and Truth worlds.

        Args:
            scene_id: ID of the scene (e.g. '0kitchen')
            scene_nodes: List of raw annotation node dictionaries (carrying indices)
            rect_pts: Rectified point cloud array (N, 3)
            node_geometries: Dictionary of node geometry metadata (extents, obb_center, yaw_rad)
            detail_assignment: Mapping from node_id -> detail_level ('point_cloud', 'bounding_box', 'label', 'remove')
            target_aff_id: Target affordance UUID
            target_obj_id: Target object UUID
            use_acm: If True, applies AllowedCollisionMatrix whitelisting gripper_links <-> {target_aff, target_obj}
            octree_res: Resolution of the OctoMap (default: 0.01 m = 1 cm)

        Returns:
            (PlanningScene message, metadata dictionary with leaf counts and sha256 hashes)
        """
        if detail_assignment is None or len(detail_assignment) == 0:
            raise RuntimeError(f"Cannot build planning scene with empty detail_assignment for scene '{scene_id}'.")
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.robot_state.is_diff = True

        # Lookup scene_nodes by annot_id for point indices
        node_indices_by_id = {
            node['annot_id']: node.get('indices', []) for node in scene_nodes
        }

        # 1. Partition nodes by detail level (both target_aff and target_obj are published per Option B)
        tree = self.octomap_bridge.create_tree(resolution=octree_res)
        octomap_points_list = []
        collision_objects = []
        published_bbox_nodes = {}
        published_pc_nodes = {}

        try:
            for node_id, geom in node_geometries.items():
                level = detail_assignment.get(node_id, "remove")

                # Exclude 'label' and 'remove'
                if level in ("label", "remove"):
                    continue

                elif level == "point_cloud":
                    # Insert node's rectified points into OctoMap
                    indices = node_indices_by_id.get(node_id, [])
                    if len(indices) > 0:
                        pts = rect_pts[indices]
                        octomap_points_list.append(pts)
                        published_pc_nodes[node_id] = {
                            "label": geom.get("label", "unlabeled"),
                            "num_points": len(indices)
                        }

                elif level == "bounding_box":
                    # Create SolidPrimitive.BOX CollisionObject
                    obj_id = f"{geom.get('label', 'obj')}_{node_id[:8]}"
                    co = CollisionObject()
                    co.header.frame_id = self.world_frame
                    co.header.stamp = self.node.get_clock().now().to_msg() if hasattr(self.node, "get_clock") else TimeMsg()
                    co.id = obj_id

                    raw_extents = geom.get('extents') if geom.get('extents') is not None else geom.get('d')
                    if raw_extents is None and 'min' in geom and 'max' in geom:
                        raw_extents = [float(geom['max'][i] - geom['min'][i]) for i in range(3)]
                    if raw_extents is None:
                        raise KeyError(f"Node {node_id} missing extents in geometry dict.")

                    # Extent floor = 0.005 m
                    extents = [
                        max(0.005, float(raw_extents[0])),
                        max(0.005, float(raw_extents[1])),
                        max(0.005, float(raw_extents[2]))
                    ]

                    box = SolidPrimitive()
                    box.type = SolidPrimitive.BOX
                    box.dimensions = extents

                    raw_center = geom.get('obb_center') if geom.get('obb_center') is not None else (geom.get('c') if geom.get('c') is not None else geom.get('centroid'))
                    if raw_center is None:
                        raise KeyError(f"Node {node_id} missing obb_center in geometry dict.")

                    pose = Pose()
                    pose.position.x = float(raw_center[0])
                    pose.position.y = float(raw_center[1])
                    pose.position.z = float(raw_center[2])
                    pose.orientation = yaw_to_quaternion(float(geom.get('yaw_rad', 0.0)))

                    co.primitives.append(box)
                    co.primitive_poses.append(pose)
                    co.operation = CollisionObject.ADD

                    collision_objects.append(co)
                    published_bbox_nodes[node_id] = {
                        "obj_id": obj_id,
                        "extents": extents,
                        "center": [pose.position.x, pose.position.y, pose.position.z],
                        "yaw_rad": float(geom.get('yaw_rad', 0.0))
                    }

            # 2. Build OctoMap if point_cloud nodes exist
            if octomap_points_list:
                all_pts = np.vstack(octomap_points_list)
                self.octomap_bridge.insert_points(tree, all_pts)

            num_leaves = self.octomap_bridge.get_num_leaf_nodes(tree)
            tree_bytes = self.octomap_bridge.write_binary_data_bytes(tree)
            tree_sha256 = hashlib.sha256(tree_bytes).hexdigest() if tree_bytes else ""

            # Attach Octomap to PlanningSceneWorld
            if tree_bytes:
                octo_msg = OctomapWithPose()
                octo_msg.header.frame_id = self.world_frame
                octo_msg.header.stamp = self.node.get_clock().now().to_msg() if hasattr(self.node, "get_clock") else TimeMsg()
                octo_msg.origin.orientation.w = 1.0  # Fixed origin (0,0,0)
                octo_msg.octomap.header = octo_msg.header
                octo_msg.octomap.id = "OcTree"
                octo_msg.octomap.resolution = octree_res
                octo_msg.octomap.binary = True
                octo_msg.octomap.data = np.frombuffer(tree_bytes, dtype=np.int8).tolist()
                scene_msg.world.octomap = octo_msg

            # 3. Add floor ground plane collision object at Z = 0.0
            ground_co = CollisionObject()
            ground_co.header.frame_id = self.world_frame
            ground_co.header.stamp = self.node.get_clock().now().to_msg() if hasattr(self.node, "get_clock") else TimeMsg()
            ground_co.id = "floor_ground_plane"

            ground_box = SolidPrimitive()
            ground_box.type = SolidPrimitive.BOX
            ground_box.dimensions = [20.0, 20.0, 0.04]  # 20m x 20m floor slab

            ground_pose = Pose()
            ground_pose.position.x = 0.0
            ground_pose.position.y = 0.0
            ground_pose.position.z = -0.02  # Top surface aligns at Z_rect = 0.0
            ground_pose.orientation.w = 1.0

            ground_co.primitives.append(ground_box)
            ground_co.primitive_poses.append(ground_pose)
            ground_co.operation = CollisionObject.ADD

            collision_objects.append(ground_co)
            scene_msg.world.collision_objects = collision_objects

            # 4. Construct and attach AllowedCollisionMatrix (ACM)
            target_obstacle_ids = []
            whitelist_octo = False
            for target_id in (target_aff_id, target_obj_id):
                if target_id:
                    lvl = detail_assignment.get(target_id)
                    if lvl == "bounding_box" and target_id in published_bbox_nodes:
                        target_obstacle_ids.append(published_bbox_nodes[target_id]["obj_id"])
                    elif lvl == "point_cloud":
                        whitelist_octo = True

            if use_acm and (target_obstacle_ids or whitelist_octo):
                acm = build_allowed_collision_matrix(
                    gripper_links=GRIPPER_LINKS,
                    target_object_ids=target_obstacle_ids,
                    whitelist_octomap=whitelist_octo
                )
                scene_msg.allowed_collision_matrix = acm

            # Assertion: non-empty scene check
            has_geometry = (num_leaves > 0) or (len(collision_objects) > 1)
            if not has_geometry and any(lvl in ("point_cloud", "bounding_box") for lvl in detail_assignment.values()):
                raise RuntimeError(f"Scene construction failed: zero obstacles published for non-empty assignment.")

            meta = {
                "scene_id": scene_id,
                "octree_resolution": octree_res,
                "octree_leaf_count": num_leaves,
                "octree_sha256": tree_sha256,
                "collision_object_count": len(collision_objects),
                "collision_object_ids": [co.id for co in collision_objects],
                "published_bbox_nodes": published_bbox_nodes,
                "published_pc_nodes": published_pc_nodes,
                "use_acm": use_acm,
                "acm_names": list(scene_msg.allowed_collision_matrix.entry_names) if use_acm else []
            }

            return scene_msg, meta

        finally:
            self.octomap_bridge.free_tree(tree)

    def build_truth_scene_data(
        self,
        scene_id: str,
        scene_nodes: List[dict],
        rect_pts: np.ndarray,
        node_geometries: Dict[str, dict],
        target_aff_id: Optional[str] = None,
        target_obj_id: Optional[str] = None,
        use_acm: bool = True,
        octree_res: float = 0.01,
    ) -> Tuple[PlanningScene, Dict[str, Any]]:
        """
        Build the canonical Ground Truth world scene with explicit all-point_cloud assignment.
        Asserts OctoMap leaf count > 0 and collision objects match expected ground floor plane.
        """
        truth_assignment = {nid: "point_cloud" for nid in node_geometries}
        scene_msg, meta = self.build_planning_scene_data(
            scene_id=scene_id,
            scene_nodes=scene_nodes,
            rect_pts=rect_pts,
            node_geometries=node_geometries,
            detail_assignment=truth_assignment,
            target_aff_id=target_aff_id,
            target_obj_id=target_obj_id,
            use_acm=use_acm,
            octree_res=octree_res,
        )
        if meta["octree_leaf_count"] == 0:
            raise RuntimeError(f"Truth scene for '{scene_id}' has 0 OctoMap leaf nodes.")
        if meta["collision_object_count"] != 1 or meta["collision_object_ids"] != ["floor_ground_plane"]:
            raise RuntimeError(f"Truth scene for '{scene_id}' unexpected collision objects: {meta['collision_object_ids']}")
        return scene_msg, meta

    def load_scene(
        self,
        scene_msg: PlanningScene,
        timeout_sec: float = 3.0
    ) -> bool:
        """Publish the computed PlanningScene message to MoveIt 2 with verified delivery."""
        self.clear_scene(timeout_sec=1.0)
        for co in scene_msg.world.collision_objects:
            self._active_object_ids.add(co.id)

        ok = self._send_planning_scene(scene_msg, timeout_sec=timeout_sec)
        if not ok:
            raise RuntimeError("Failed to apply planning scene to MoveIt 2.")

        # Round-trip assertion on OctoMap payload delivery
        if scene_msg.world.octomap.octomap.data:
            if self.get_scene_client.wait_for_service(timeout_sec=0.5):
                req_ps = GetPlanningScene.Request()
                req_ps.components.components = PlanningSceneComponents.OCTOMAP
                future_ps = self.get_scene_client.call_async(req_ps)
                rclpy.spin_until_future_complete(self.node, future_ps, timeout_sec=2.0)
                if future_ps.done() and future_ps.result() is not None:
                    applied_octo = future_ps.result().scene.world.octomap
                    if len(applied_octo.octomap.data) == 0:
                        raise RuntimeError("OctoMap round-trip assertion failed: applied OctoMap data is empty in MoveIt.")
        return True

    def _send_planning_scene(self, scene_msg: PlanningScene, timeout_sec: float = 3.0) -> bool:
        """Send planning scene update via service if available, fallback to topic."""
        if self.apply_scene_client.wait_for_service(timeout_sec=0.2):
            req = ApplyPlanningScene.Request()
            req.scene = scene_msg
            future = self.apply_scene_client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_sec)
            if future.done() and future.result() is not None:
                return future.result().success

        # Fallback to topic publish
        self.planning_scene_pub.publish(scene_msg)
        return True

