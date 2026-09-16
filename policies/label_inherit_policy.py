"""label_inherit: score objects on their node label, expand their sub-parts.

Signal is the node LABEL alone, matched against the task with the shared
embedder; that similarity is the object score. The shared object step
(assign_objects) buckets the objects and anchors the top one at POINT_CLOUD;
sub-parts then inherit their parent via resolve_subparts.

Propagation is parent -> sub-part inheritance. The affordance string itself is
never read, the edge serving purely as structure, which makes this the
label-only control arm of the label / affordance / combined ablation: it sees
exactly the same graph as affordance_inherit but none of its natural-language
affordance text.
"""

from policies.scene_graph import SceneGraph, PolicyResult
from policies.utils.text_embedder import get_embedder
from policies.utils.relevance_scoring import (
    assign_objects,
    resolve_subparts,
)


def apply_label_inherit_policy(graph: SceneGraph, task: str) -> PolicyResult:
    """Assign detail levels by scoring objects on their label similarity to the task."""
    node_ids = [n.id for n in graph.nodes]
    if not node_ids:
        return PolicyResult("label_inherit", task, {})

    embedder = get_embedder()
    sims = embedder.similarity([n.label for n in graph.nodes], [task])[:, 0]
    score = {nid: float(s) for nid, s in zip(node_ids, sims)}

    assignment = assign_objects(node_ids, graph, score)
    resolve_subparts(assignment, graph)
    return PolicyResult("label_inherit", task, assignment)
