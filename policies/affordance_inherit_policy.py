"""affordance_inherit: relevance from the natural-language affordance strings.

Signal is the AFFORDANCE text on the edges ("rotate to adjust the temperature",
"push or press to open") rather than the node labels. Each edge's cosine score
to the task is propagated to the parent object, so an object's score is the best
of its children's affordance edges; an object with no affordance edge falls back
to a discounted node-label similarity. Object levels are bucketed from that
score.

Propagation is parent -> sub-part inheritance, the same shared rule
(resolve_subparts) the other inheriting policies use. Scoring the affordance
text lets a task light up the right object even when its noun is nowhere near
the task wording, as in "make the room warmer" reaching a radiator through
"rotate to adjust the temperature". This is the affordance arm of the label /
affordance / combined ablation.
"""

from policies.scene_graph import SceneGraph, PolicyResult
from policies.utils.text_embedder import get_embedder
from policies.utils.relevance_scoring import (
    assign_objects,
    resolve_subparts,
)

# Weight on the node-label fallback for objects that carry no affordance edge.
_LABEL_FALLBACK_WEIGHT = 0.8


def apply_affordance_inherit_policy(graph: SceneGraph, task: str) -> PolicyResult:
    """Assign detail levels from per-edge affordance similarity to the task."""
    node_ids = [n.id for n in graph.nodes]
    if not node_ids:
        return PolicyResult("affordance_inherit", task, {})

    embedder = get_embedder()
    label_sims = dict(zip(
        node_ids,
        embedder.similarity([n.label for n in graph.nodes], [task])[:, 0],
    ))

    # Best affordance-edge score reaching each node (as sub-part or as parent).
    edge_score = {nid: None for nid in node_ids}
    if graph.edges:
        e_sims = embedder.similarity([e.label for e in graph.edges], [task])[:, 0]
        for edge, s in zip(graph.edges, e_sims):
            for nid in (edge.source_id, edge.target_id):
                if nid in edge_score:
                    prev = edge_score[nid]
                    edge_score[nid] = float(s) if prev is None else max(prev, float(s))

    # Object score: the affordance edge if any, else a discounted label similarity.
    score = {
        nid: (edge_score[nid] if edge_score[nid] is not None
              else _LABEL_FALLBACK_WEIGHT * float(label_sims[nid]))
        for nid in node_ids
    }

    assignment = assign_objects(node_ids, graph, score)
    resolve_subparts(assignment, graph)
    return PolicyResult("affordance_inherit", task, assignment)
