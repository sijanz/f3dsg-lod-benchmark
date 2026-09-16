"""Internal scene graph data model."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json


# Edge labels with structural (non-affordance) meaning. Edges carry no type
# field, so the label string is the only discriminator; these must match the
# `description` values written by scripts/fuse_scenes.py.
CONTAINS_EDGE_LABEL = "contains"    # ROOM -> object containment
REACHABLE_EDGE_LABEL = "reachable"  # ROOM <-> ROOM robot reachability
# ROOM -> passage object (door/elevator/...) mediating a reachable room pair.
# Direction ROOM -> object on purpose (same reason as `contains`)
PASSAGE_EDGE_LABEL = "passage"

# Stand-in for an affordance edge label in the ablation arms that withhold the
# natural-language affordance text (llm_zero_shot_noaff, llm_remote_haiku_noaff,
# information_bottleneck_noaff). Those arms keep the graph structure intact and
# replace only the text, so "handle --pull to open--> microwave oven" becomes
# "handle --part of--> microwave oven". Defining it once is what makes the arms
# comparable across policies: each of them withholds exactly the same thing.
NEUTRAL_RELATION_LABEL = "part of"


class NodeLevel(Enum):
    """Semantic hierarchy level of a node.

    Set by dataset converters. Nodes from datasets without a hierarchy
    (e.g. FunGraph3D) leave this as None.
    """
    BUILDING = "building"
    ROOM     = "room"
    OBJECT   = "object"


@dataclass
class Node:
    id: str
    label: str
    position: Optional[List[float]] = None
    indices: Optional[List[int]] = None
    level: Optional[NodeLevel] = None
    obb: Optional[dict] = None  # {"center":[x,y,z], "R":[[3x3]], "extent":[ex,ey,ez]}


@dataclass
class Edge:
    source_id: str
    target_id: str
    label: str


@dataclass
class SceneGraph:
    scene_id: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def node_ids(self) -> set:
        return {n.id for n in self.nodes}

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edges_from_node(self, source_id: str) -> List[Edge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source_id == source_id]

    def get_edges_to_node(self, target_id: str) -> List[Edge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target_id == target_id]

    def nodes_with_position_count(self) -> int:
        """Count nodes that have a valid position."""
        return sum(1 for n in self.nodes if n.position is not None)

    def get_room_adjacency(self) -> Dict[str, set]:
        """Robot reachability between ROOM nodes, as an undirected adjacency map.

        Maps every ROOM node id to the ids of the rooms it is mutually
        reachable with (via REACHABLE_EDGE_LABEL edges). Rooms without such
        an edge map to an empty set: the robot cannot move between them and
        any other room. Empty dict for datasets without ROOM nodes.
        """
        room_ids = {n.id for n in self.nodes if n.level == NodeLevel.ROOM}
        adjacency = {rid: set() for rid in room_ids}
        for e in self.edges:
            if (e.label == REACHABLE_EDGE_LABEL
                    and e.source_id in room_ids and e.target_id in room_ids):
                adjacency[e.source_id].add(e.target_id)
                adjacency[e.target_id].add(e.source_id)
        return adjacency

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "position": n.position,
                    "obb": n.obb,
                    "indices": n.indices,
                    "level": n.level.value if n.level is not None else None,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "label": e.label,
                }
                for e in self.edges
            ],
    }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneGraph":
        nodes = [
            Node(
                id=n["id"],
                label=n["label"],
                position=n.get("position"),
                obb=n.get("obb"),
                indices=n.get("indices"),
                level=NodeLevel(n["level"]) if n.get("level") else None,
            )
            for n in data["nodes"]
        ]

        edges = [
            Edge(
                source_id=e["source_id"],
                target_id=e["target_id"],
                label=e["label"],
            )
            for e in data["edges"]
        ]

        return cls(
            scene_id=data["scene_id"],
            nodes=nodes,
            edges=edges,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SceneGraph":
        return cls.from_dict(json.loads(json_str))


@dataclass
class PolicyContext:
    """Per-task framework context. Every task has a pose.

    Lives at the registry-wrapper boundary only: policies themselves never
    receive it. Built by GenericPolicyNode from the GraphBuilder payload.
    """

    robot_pose: List[float]  # [x, y, z]
    start_room_id: Optional[str] = None  # None on flat datasets


class DetailLevel(Enum):
    """Detail level for node representation in the final graph."""
    
    REMOVE = "remove"  # Node is completely removed
    LABEL = "label"  # Only label and position
    BOUNDING_BOX = "bounding_box"  # Label, position, and bounding box
    POINT_CLOUD = "point_cloud"  # Full point cloud representation


@dataclass
class PolicyResult:
    """Result from applying a policy to a scene graph.

    Maps each node ID to a detail level indicating how much detail
    should be preserved for that node in the final graph. Optionally,
    a node ID can also be mapped to a merge group ID: every node sharing
    the same merge group ID is combined into a single synthetic node by
    the GraphModifier. Nodes absent from merge_assignment (the default
    for all existing policies) are left unmerged.
    """

    policy_name: str
    task: str
    detail_assignment: Dict[str, DetailLevel]  # node_id -> DetailLevel
    merge_assignment: Dict[str, str] = field(default_factory=dict)  # node_id -> merge_group_id
    # Framework-written extras (e.g. the routing wrapper's goal room, room
    # chain and feasibility live under metadata["routing"]).
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = {
            "policy_name": self.policy_name,
            "task": self.task,
            "detail_assignment": {
                node_id: detail.value
                for node_id, detail in self.detail_assignment.items()
            },
            "merge_assignment": dict(self.merge_assignment),
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyResult":
        """Create from dictionary."""
        return cls(
            policy_name=data["policy_name"],
            task=data["task"],
            detail_assignment={
                node_id: DetailLevel(detail_str)
                for node_id, detail_str in data["detail_assignment"].items()
            },
            merge_assignment=dict(data.get("merge_assignment", {})),
            metadata=dict(data.get("metadata", {})),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PolicyResult":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))