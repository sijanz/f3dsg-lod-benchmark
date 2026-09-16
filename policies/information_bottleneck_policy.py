"""Information Bottleneck resolution policy over functional scene graphs.

Thin binding of the shared IB core (utils/ib_common) to the full condition,
in which a node's task relevance may be established through its own label or
through the natural-language affordance text on any of its edges.

Task-driven Agglomerative Information Bottleneck (Slonim & Tishby, 1999) in
the form CLIO restates as its Algorithm 1 (Maggio et al., 2024). The
modelling layer, and every deliberate deviation from CLIO with its reason,
is documented in utils/ib_common.
"""

from policies.scene_graph import SceneGraph, PolicyResult
from policies.utils.ib_common import apply_ib


def apply_information_bottleneck_policy(
    graph: SceneGraph, task: str
) -> PolicyResult:
    """Cluster the graph via Agglomerative IB, affordance text included."""
    return apply_ib(graph, task, "information_bottleneck", affordances=True)
