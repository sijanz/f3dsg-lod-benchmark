"""combined_quantile: combined_inherit with fixed rank fractions instead of thresholds.

Signal and propagation are identical to combined_inherit. The objects are scored
by the very same function (combined_score) and the sub-parts inherit through the
very same rule (resolve_subparts). The one difference is the cut rule: objects
are ranked among themselves and cut at fixed fractions rather than compared
against absolute cosine values.

That single difference is what the policy exists to measure. Cosine similarity
has no scene-independent scale, so hand-set thresholds are the standard
objection to this kind of policy; ranking answers it, and comparing this arm to
combined_inherit says whether the absolute values were contributing anything
beyond the ordering they induce.

Its own assumption is visible in the failure mode. Fixed fractions keep a
constant share of every scene, so what is retained grows with scene size, while
the ground truth keeps a roughly constant absolute number of objects at full
detail no matter how large the scene is. Large scenes are therefore where this
arm should lose.
"""

from policies.scene_graph import SceneGraph, PolicyResult
from policies.utils.relevance_scoring import (
    assign_objects_quantile,
    combined_score,
    resolve_subparts,
)


def apply_combined_quantile_policy(graph: SceneGraph, task: str) -> PolicyResult:
    """Assign detail levels by cutting the combined relevance ranking at fixed fractions."""
    node_ids = [n.id for n in graph.nodes]
    if not node_ids:
        return PolicyResult("combined_quantile", task, {})

    score = combined_score(graph, task)
    assignment = assign_objects_quantile(node_ids, graph, score)
    resolve_subparts(assignment, graph)
    return PolicyResult("combined_quantile", task, assignment)
