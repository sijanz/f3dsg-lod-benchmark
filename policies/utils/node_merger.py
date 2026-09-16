"""Combine several full-detail nodes into one synthetic merged node.

Only meaningful on the full graph (where obb/indices may be populated),
since the minimal graph handed to policies never carries that data. Used
by GraphModifierNode whenever a PolicyResult's merge_assignment groups
two or more node IDs together.
"""

from enum import Enum
from typing import List, Optional

import numpy as np

from policies.scene_graph import DetailLevel, Node

# Ordering used to compare DetailLevel values for the MAX/MIN merge strategies.
_DETAIL_LEVEL_RANK = {
    DetailLevel.REMOVE: 0,
    DetailLevel.LABEL: 1,
    DetailLevel.BOUNDING_BOX: 2,
    DetailLevel.POINT_CLOUD: 3,
}


class MergeDetailStrategy(Enum):
    """How to resolve a single DetailLevel for a merged node from its members."""

    MAX = "max"
    MIN = "min"
    FIXED_BOUNDING_BOX = "fixed_bounding_box"


def resolve_merged_detail_level(
    levels: List[DetailLevel],
    strategy: MergeDetailStrategy = MergeDetailStrategy.MAX,
) -> DetailLevel:
    """Resolve the DetailLevel of a merged node from its members' levels.

    Default is MAX: a merge can only reduce node count, never drop detail
    below what any individual member already earned.
    """
    if not levels:
        return DetailLevel.REMOVE

    if strategy == MergeDetailStrategy.FIXED_BOUNDING_BOX:
        return DetailLevel.BOUNDING_BOX
    elif strategy == MergeDetailStrategy.MIN:
        return min(levels, key=lambda lvl: _DETAIL_LEVEL_RANK[lvl])
    else:
        return max(levels, key=lambda lvl: _DETAIL_LEVEL_RANK[lvl])


def _pick_representative_label(members: List[Node], levels: List[DetailLevel]) -> str:
    """Pick the label of the most relevant member (highest individual DetailLevel)."""
    best_idx = max(
        range(len(members)), key=lambda i: _DETAIL_LEVEL_RANK[levels[i]]
    )
    return members[best_idx].label


def _merge_positions(members: List[Node]) -> List[float]:
    positions = [m.position for m in members if m.position is not None]
    dims = len(positions[0])
    return [
        sum(p[i] for p in positions) / len(positions)
        for i in range(dims)
    ]


def _merge_obb(members: List[Node]) -> Optional[dict]:
    if any(m.obb is None for m in members):
        return None
    try:
        import open3d as o3d
        signs = np.array([[s0, s1, s2]
                          for s0 in [-1, 1]
                          for s1 in [-1, 1]
                          for s2 in [-1, 1]], dtype=float)
        all_corners = []
        for m in members:
            center = np.array(m.obb["center"])
            R = np.array(m.obb["R"])
            half = np.array(m.obb["extent"]) / 2.0
            corners = center + (R @ (signs * half).T).T
            all_corners.append(corners)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(all_corners))
        obb = pcd.get_oriented_bounding_box()
        return {
            "center": np.asarray(obb.center).tolist(),
            "R":      np.asarray(obb.R).tolist(),
            "extent": np.asarray(obb.extent).tolist(),
        }
    except Exception:
        return None


def _merge_indices(members: List[Node]):
    if any(m.indices is None for m in members):
        return None
    merged = []
    seen = set()
    for m in members:
        for idx in m.indices:
            if idx not in seen:
                seen.add(idx)
                merged.append(idx)
    return merged


def merge_nodes(
    members: List[Node],
    new_id: str,
    levels: List[DetailLevel] = None,
) -> Node:
    """Combine multiple full Nodes into one synthetic Node.

    - position: centroid of member positions.
    - indices: deduplicated union of member point indices (None if any member lacks them).
    - label: label of the most relevant member (per `levels`, defaulting to
      the member order when levels aren't provided).
    """
    if levels is None:
        levels = [DetailLevel.LABEL] * len(members)

    return Node(
        id=new_id,
        label=_pick_representative_label(members, levels),
        position=_merge_positions(members),
        obb=_merge_obb(members),
        indices=_merge_indices(members),
        level=members[0].level,
    )
