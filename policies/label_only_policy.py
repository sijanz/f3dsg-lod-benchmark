"""label_only: the structure-blind semantic baseline.

Signal is the node LABEL, matched against the task with the shared embedder.
Propagation is none: every node's detail level is bucketed from its own
similarity alone, with no edges, no seeds spread through the graph and no
parent/child coupling.

It is deliberately the structure-blind twin of label_inherit. Objects are scored
with the exact same shared step (assign_objects, same thresholds and same
POINT_CLOUD anchor), but sub-parts are judged by their OWN label instead of
inheriting their parent, which is to say it skips resolve_subparts. That missing
structural step is the single variable between the two, so comparing label_only
to label_inherit isolates the value of exploiting the parent->child structure.

The comparison cuts both ways. A task that names a sub-part rather than the
object it belongs to, as in "turn the knob", is the case where inheritance drags
the part down to its parent's level and judging the part by its own label wins.
"""

from policies.scene_graph import SceneGraph, PolicyResult
from policies.utils.text_embedder import get_embedder
from policies.utils.relevance_scoring import (
    assign_objects,
    bucket,
    is_sub_part,
)


def apply_label_only_policy(graph: SceneGraph, task: str) -> PolicyResult:
    """Assign detail levels by per-node cosine similarity between labels and task."""
    node_ids = [n.id for n in graph.nodes]
    if not node_ids:
        return PolicyResult("label_only", task, {})

    sims = get_embedder().similarity([n.label for n in graph.nodes], [task])[:, 0]
    score = {nid: float(s) for nid, s in zip(node_ids, sims)}

    # Objects: scored exactly like the inheriting policies (shared thresholds + anchor).
    assignment = assign_objects(node_ids, graph, score)
    # Sub-parts: judged by their OWN label, with no parent inheritance. This missing
    # structural step is the single variable versus label_inherit.
    for nid in node_ids:
        if is_sub_part(graph, nid):
            assignment[nid] = bucket(score[nid])

    return PolicyResult("label_only", task, assignment)
