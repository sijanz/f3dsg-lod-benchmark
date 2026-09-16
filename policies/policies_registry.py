"""
Central configuration for all available policies.

When adding a new policy:
1. Create the policy module: policies/{policy_name}_policy.py
2. Add an entry to POLICIES below, with the full module path

Naming conventions (enforced by get_policy_module_and_function):
- Policy module: policies/{policy_name}_policy.py
- Policy function: apply_{policy_name}_policy()

The order of POLICIES is the canonical order results appear in evaluation.json,
keeping ablation partners adjacent.
"""

import importlib

from policies.scene_graph import DetailLevel, NodeLevel

POLICIES: dict[str, dict[str, str]] = {
    # Trivial baseline: keeps everything, the reference for memory savings.
    "identity": {
        "module": "policies.identity_policy",
        "function": "apply_identity_policy",
    },
    # Embedding policies. The name says what the policy matches against the
    # task (label / affordance / combined) and how it spreads detail through
    # the scene (only = nothing, inherit = parent -> sub-part, radius =
    # Euclidean).
    "label_only": {
        "module": "policies.label_only_policy",
        "function": "apply_label_only_policy",
    },
    "label_inherit": {
        "module": "policies.label_inherit_policy",
        "function": "apply_label_inherit_policy",
    },
    "affordance_inherit": {
        "module": "policies.affordance_inherit_policy",
        "function": "apply_affordance_inherit_policy",
    },
    # The three cut-rule arms over one and the same combined score: absolute
    # cosine threshold, fixed rank fraction, largest gap in the ranking.
    "combined_inherit": {
        "module": "policies.combined_inherit_policy",
        "function": "apply_combined_inherit_policy",
    },
    "combined_quantile": {
        "module": "policies.combined_quantile_policy",
        "function": "apply_combined_quantile_policy",
    },
    "combined_quantile_adaptive": {
        "module": "policies.combined_quantile_adaptive_policy",
        "function": "apply_combined_quantile_adaptive_policy",
    },
    "label_radius": {
        "module": "policies.label_radius_policy",
        "function": "apply_label_radius_policy",
    },
    # Algorithmic / structural information-theoretic policies.
    "information_bottleneck": {
        "module": "policies.information_bottleneck_policy",
        "function": "apply_information_bottleneck_policy",
    },
    "information_bottleneck_noaff": {
        "module": "policies.information_bottleneck_noaff_policy",
        "function": "apply_information_bottleneck_noaff_policy",
    },
}

ALL_POLICIES: list[str] = list(POLICIES.keys())

# Result-name suffix for the routed condition ({policy}+routed), so plain
# and routed runs of the same policy coexist in evaluation.json.
ROUTED_SUFFIX = "+routed"


def get_policy_module_and_function(policy_name: str) -> tuple[str, str]:
    if policy_name not in POLICIES:
        available = ", ".join(ALL_POLICIES)
        raise KeyError(
            f"Unknown policy '{policy_name}'. Available: {available}"
        )
    policy_info = POLICIES[policy_name]
    module = policy_info["module"]
    function = policy_info["function"]

    # Module must end with {policy_name}_policy; the package prefix is free.
    expected_module_suffix = f"{policy_name}_policy"
    expected_function = f"apply_{policy_name}_policy"
    errors = []
    if not module.endswith(expected_module_suffix):
        errors.append(
            f"  module: expected to end with '{expected_module_suffix}', got '{module}'"
        )
    if function != expected_function:
        errors.append(
            f"  function: expected '{expected_function}', got '{function}'"
        )
    if errors:
        raise ValueError(
            f"Registry entry for '{policy_name}' violates naming convention:\n"
            + "\n".join(errors)
        )

    return module, function


def _pin_rooms_to_label(result, graph):
    """Framework room invariant: ROOM nodes are structural, so policies do
    not decide their detail level.

    Pin every ROOM node to LABEL (so rooms always survive into the result /
    visualization) and drop them from merge groups. Runs on the full graph
    after the policy; no-op on flat datasets (level is None). Mutates
    ``result`` in place.
    """
    for node in graph.nodes:
        if node.level == NodeLevel.ROOM:
            result.detail_assignment[node.id] = DetailLevel.LABEL
            result.merge_assignment.pop(node.id, None)


def _run_with_routing(policy_fn, graph, task, context):
    """Multi-room decomposition: run a single-room policy multi-room."""
    try:
        from policies.utils.room_routing import build_room_subgraph, plan_route
    except ImportError:
        raise NotImplementedError("Multi-room routing utility is not included in this benchmark release.")

    plan = plan_route(graph, task, context)
    if plan is None:
        result = policy_fn(graph, task)
    else:
        subgraph = build_room_subgraph(graph, plan.goal_member_ids)
        result = policy_fn(subgraph, task)
        for node_id in plan.remove_ids:
            result.detail_assignment[node_id] = DetailLevel.REMOVE
        for node_id in plan.landmark_ids:
            result.detail_assignment.setdefault(node_id, DetailLevel.LABEL)
        for node_id in plan.passage_detail_ids:
            result.detail_assignment[node_id] = DetailLevel.POINT_CLOUD
            result.merge_assignment.pop(node_id, None)
        routing = {
            "goal_room_id": plan.goal_room_id,
            "room_path": plan.path,
            "passage_ids": list(plan.passage_detail_ids),
            "feasible": plan.feasible,
        }
        if plan.infeasibility_reason is not None:
            routing["infeasibility_reason"] = plan.infeasibility_reason
        result.metadata["routing"] = routing
    result.policy_name = f"{result.policy_name}{ROUTED_SUFFIX}"
    return result


def load_policy_function(policy_name: str, routing: bool = False):
    """Import a policy and wrap it with the framework invariants.

    Every consumer (pipeline nodes, tests) must load policies through this
    function; importing ``apply_*`` directly bypasses the room invariant.
    The returned callable has the ``(graph, task, context=None)`` signature
    and runs a fixed pipeline: optionally the multi-room routing
    decomposition (``routing=True``), then the room invariant. Inner
    policies keep the plain ``apply_*(graph, task)`` signature and never see
    pose/room data — room logic is framework business.
    """
    module_path, function_name = get_policy_module_and_function(policy_name)
    module = importlib.import_module(module_path)
    raw = getattr(module, function_name)

    def run(graph, task, context=None):
        if routing:
            result = _run_with_routing(raw, graph, task, context)
        else:
            result = raw(graph, task)
        _pin_rooms_to_label(result, graph)
        return result
    return run
