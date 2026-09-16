"""combined_quantile_adaptive: cut the combined relevance ranking at its own gaps.

Signal and propagation are identical to combined_inherit and combined_quantile:
the same combined_score over node labels and affordance edges, the same
sub-part inheritance. Only the cut rule differs. The objects are ranked and the
three largest drops between consecutive scores become the level boundaries, so
the scene itself decides where the levels lie.

The three arms form a ladder over what a cut rule is allowed to assume.
combined_inherit assumes an absolute cosine value, combined_quantile assumes a
proportion of the scene, and this arm assumes only that a scene separates
relevant from irrelevant objects by a visible margin. It is accordingly the only
one of the three that carries no hand-set constant at all, which is the direct
answer to the objection that cosine thresholds are calibrated by hand and do not
transfer between scenes or embedders.

The cost is a dependence on the shape of the score distribution rather than on
its values. A scene whose objects are all similarly relevant offers no clear
gap, and the rule then cuts wherever the largest of several small drops happens
to fall. Like combined_quantile it carries no absolute floor, so its top group
is promoted whatever it scored.
"""

from policies.scene_graph import SceneGraph, PolicyResult
from policies.utils.relevance_scoring import (
    assign_objects_adaptive,
    combined_score,
    resolve_subparts,
)


def apply_combined_quantile_adaptive_policy(graph: SceneGraph, task: str) -> PolicyResult:
    """Assign detail levels by cutting the combined relevance ranking at its largest gaps."""
    node_ids = [n.id for n in graph.nodes]
    if not node_ids:
        return PolicyResult("combined_quantile_adaptive", task, {})

    score = combined_score(graph, task)
    assignment = assign_objects_adaptive(node_ids, graph, score)
    resolve_subparts(assignment, graph)
    return PolicyResult("combined_quantile_adaptive", task, assignment)
