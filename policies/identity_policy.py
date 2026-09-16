""" Identity policy for demonstration purposes. Replaces original graph logic """

from policies.scene_graph import SceneGraph, PolicyResult, DetailLevel

def apply_identity_policy(graph: SceneGraph, task: str) -> PolicyResult:
    """
    Identity policy: keeps all nodes and edges with full detail.
    
    This is a simple baseline that does not perform any filtering or abstraction.
    
    Args:
        graph: Full scene graph
        task: Task description string (ignored in this policy)
        
    Returns:
        PolicyResult with all nodes assigned POINT_CLOUD detail
    """
    detail_assignment = {
        node.id: DetailLevel.POINT_CLOUD
        for node in graph.nodes
    }
    
    return PolicyResult(
        policy_name="identity",
        task=task,
        detail_assignment=detail_assignment,
    )