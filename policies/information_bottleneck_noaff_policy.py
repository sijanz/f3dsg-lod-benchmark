"""Information Bottleneck policy without the natural-language affordance text.

Thin binding of the shared IB core (utils/ib_common) to the withheld
condition. The graph structure stays fully intact and only the edge labels
are replaced by the neutral relation, so a node is still scored against its
neighbour's identity ("part of the microwave oven") but no longer against
what the connection affords ("pull to open or close the microwave oven").
Dropping the edges instead would remove structure and text together and the
two effects could no longer be told apart.

The pair isolates the contribution of the affordance annotations to the IB
clustering itself: identical primitives, identical algorithm, identical
parameters, one signal removed.
"""

from policies.scene_graph import SceneGraph, PolicyResult
from policies.utils.ib_common import apply_ib


def apply_information_bottleneck_noaff_policy(
    graph: SceneGraph, task: str
) -> PolicyResult:
    """Cluster the graph via Agglomerative IB, affordance text withheld."""
    return apply_ib(
        graph, task, "information_bottleneck_noaff", affordances=False
    )
