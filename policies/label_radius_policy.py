"""label_radius: structure-blind, position-based detail spreading.

Signal is the node LABEL, used only to pick the seeds (label <-> task cosine,
the same threshold the other embedding policies use). Propagation is neither
inheritance nor bucketing but a Euclidean radius around those seeds:

  <= _RADIUS_POINT_CLOUD  m  -> point_cloud
  <= _RADIUS_BOUNDING_BOX m  -> bounding_box
  <= _RADIUS_LABEL        m  -> label
  farther (or no position)   -> remove

Because a node keeps the level of its nearest seed, closer means more detail,
which realises "keep the highest level assigned across all seeds". Nodes beyond
the label radius are removed. A seed sits at distance 0 from itself and so
reaches point_cloud, which is what normally keeps one anchor in the scene --
but only for a seed that has a position. Unlike the policies built on
assign_objects, this one carries no anchor guarantee: on a dataset whose seeds
have no position the scene comes back entirely removed, since a node without a
position can never be lifted out of the initial REMOVE.

It deliberately ignores the graph edges. That omission is the point: a
functionally related but spatially DISTANT node, a ceiling light and its wall
switch or a door and a far handle, falls outside the radius and is dropped, so
this baseline misses exactly the links the inheriting policies recover through
the structure.

The radii assume a room-scale, obb-centre coordinate frame as in FunGraph3D;
tune them for datasets at a different scale.
"""

import math

from policies.scene_graph import SceneGraph, PolicyResult, DetailLevel
from policies.utils.text_embedder import get_embedder
from policies.utils.relevance_scoring import (
    select_seed_ids,
)

_RADIUS_POINT_CLOUD = 0.75   # <= X m from a seed -> point_cloud
_RADIUS_BOUNDING_BOX = 1.5   # <= Y m -> bounding_box
_RADIUS_LABEL = 2.5          # <= Z m -> label; farther -> remove


def apply_label_radius_policy(graph: SceneGraph, task: str) -> PolicyResult:
    """Assign detail levels by distance from task-relevant seed nodes."""
    node_ids = [n.id for n in graph.nodes]
    if not node_ids:
        return PolicyResult("label_radius", task, {})

    sims = get_embedder().similarity([n.label for n in graph.nodes], [task])[:, 0]
    seeds = select_seed_ids(node_ids, sims)
    seed_positions = [n.position for n in graph.nodes
                      if n.id in seeds and n.position is not None]

    # Everything starts removed; distance to the nearest seed lifts it. Nodes beyond
    # the label radius (and nodes without a position) stay REMOVE.
    assignment = {nid: DetailLevel.REMOVE for nid in node_ids}
    for node in graph.nodes:
        if node.position is None or not seed_positions:
            continue
        nearest = min(math.dist(node.position, sp) for sp in seed_positions)
        if nearest <= _RADIUS_POINT_CLOUD:
            assignment[node.id] = DetailLevel.POINT_CLOUD
        elif nearest <= _RADIUS_BOUNDING_BOX:
            assignment[node.id] = DetailLevel.BOUNDING_BOX
        elif nearest <= _RADIUS_LABEL:
            assignment[node.id] = DetailLevel.LABEL
        # else: stays REMOVE

    return PolicyResult("label_radius", task, assignment)
