"""
Pure Python graph modification utilities for SceneGraph detail filtering and serialization.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
from policies.scene_graph import (
    SceneGraph, PolicyResult, DetailLevel, Node, Edge, NodeLevel
)
from policies.utils.node_merger import merge_nodes, resolve_merged_detail_level


def apply_detail_level(node: Node, detail: DetailLevel) -> Node:
    """Apply detail level to a node."""
    if detail == DetailLevel.LABEL:
        return Node(
            id=node.id,
            label=node.label,
            position=node.position,
            obb=node.obb if node.level == NodeLevel.ROOM else None,
            level=node.level,
        )
    elif detail == DetailLevel.BOUNDING_BOX:
        return Node(
            id=node.id,
            label=node.label,
            position=node.position,
            obb=node.obb,
            level=node.level,
        )
    elif detail == DetailLevel.POINT_CLOUD:
        return Node(
            id=node.id,
            label=node.label,
            position=node.position,
            obb=node.obb,
            level=node.level,
            point_cloud=node.point_cloud,
        )
    else:
        return None


def create_modified_graph(full_graph: SceneGraph, policy_result: PolicyResult) -> Tuple[SceneGraph, Dict[str, DetailLevel]]:
    """Create a modified graph with detail levels applied and merge groups resolved."""
    detail_assignment = policy_result.detail_assignment
    merge_assignment = policy_result.merge_assignment

    groups: Dict[str, List[str]] = defaultdict(list)
    for node in full_graph.nodes:
        group_id = merge_assignment.get(node.id)
        if group_id is not None:
            groups[group_id].append(node.id)

    merge_groups = {gid: ids for gid, ids in groups.items() if len(ids) > 1}
    merged_member_ids = {node_id for ids in merge_groups.values() for node_id in ids}

    modified_nodes: List[Node] = []
    output_detail_assignment: Dict[str, DetailLevel] = {}
    id_remap: Dict[str, str] = {}

    for node in full_graph.nodes:
        if node.id in merged_member_ids:
            continue
        detail = detail_assignment.get(node.id, DetailLevel.REMOVE)
        if detail == DetailLevel.REMOVE:
            continue
        modified_nodes.append(apply_detail_level(node, detail))
        output_detail_assignment[node.id] = detail
        id_remap[node.id] = node.id

    for group_id, member_ids in merge_groups.items():
        members = [full_graph.get_node_by_id(nid) for nid in member_ids]
        member_levels = [detail_assignment.get(nid, DetailLevel.REMOVE) for nid in member_ids]
        merged_level = resolve_merged_detail_level(member_levels)
        if merged_level == DetailLevel.REMOVE:
            continue

        merged_node = merge_nodes(members, new_id=group_id, levels=member_levels)
        modified_nodes.append(apply_detail_level(merged_node, merged_level))
        output_detail_assignment[group_id] = merged_level
        for nid in member_ids:
            id_remap[nid] = group_id

    kept_output_ids = {n.id for n in modified_nodes}
    edge_labels: Dict[Tuple[str, str], List[str]] = {}
    for edge in full_graph.edges:
        source = id_remap.get(edge.source_id)
        target = id_remap.get(edge.target_id)
        if source is None or target is None:
            continue
        if source not in kept_output_ids or target not in kept_output_ids:
            continue
        if source == target:
            continue
        labels = edge_labels.setdefault((source, target), [])
        if edge.label not in labels:
            labels.append(edge.label)

    modified_edges = [
        Edge(source_id=s, target_id=t, label=", ".join(lbls))
        for (s, t), lbls in edge_labels.items()
    ]

    modified_graph = SceneGraph(nodes=modified_nodes, edges=modified_edges)
    return modified_graph, output_detail_assignment

