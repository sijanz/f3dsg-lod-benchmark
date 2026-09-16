"""combined_inherit: node label and affordance text together, thresholded.

Signal is both, an object's relevance being the larger of its node-label
similarity and its best incident affordance-edge similarity, so a task can
select an object either by its name or by a matching affordance. Propagation is
parent -> sub-part inheritance.

This unites label_inherit's object selection with affordance_inherit's
affordance grounding and is the combined arm of the label / affordance /
combined ablation.

It is also the threshold-based arm of the cut-rule ablation family. combined_quantile and
combined_quantile_adaptive score objects with the very same function
(combined_score) and differ from this policy in nothing but how that score is
cut into the four levels, so the three of them isolate the cut rule alone.
"""

from policies.scene_graph import SceneGraph, PolicyResult
from policies.utils.relevance_scoring import (
    assign_objects,
    combined_score,
    resolve_subparts,
)


def apply_combined_inherit_policy(graph: SceneGraph, task: str) -> PolicyResult:
    """Assign detail levels from combined node-label and edge-affordance relevance."""
    node_ids = [n.id for n in graph.nodes]
    if not node_ids:
        return PolicyResult("combined_inherit", task, {})

    score = combined_score(graph, task)
    assignment = assign_objects(node_ids, graph, score)
    resolve_subparts(assignment, graph)
    return PolicyResult("combined_inherit", task, assignment)
