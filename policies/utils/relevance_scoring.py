"""Shared scoring and expansion skeleton for the embedding policies.

Not a policy module (no ``apply_*`` function) and not under ``policies/``, so
the registry can never pick it up. Holds the pieces common to label_inherit,
affordance_inherit, combined_inherit, the two quantile arms, label_only and
label_radius: sub-part / parent lookups over FunGraph3D's parent->child
affordance forest, seed selection, and the cut rules that turn a task relevance
score into a detail level.

Note that nothing here reads an affordance string. The edges are used purely as
structure, telling an object apart from a functional sub-part. Which text a
policy scores against the task is the policy's own business, and it is exactly
the variable the label / affordance / combined ablation varies.

FunGraph3D edges point object (``target_id``, parent) -> functional sub-part
(``source_id``, child: knob / handle / button), one level deep. So an object's
sub-parts are the sources of the edges pointing at it, and a node is a sub-part
iff it has an outgoing edge to a parent.
"""

from typing import Dict, List, Optional

from policies.scene_graph import SceneGraph, DetailLevel
from policies.utils.text_embedder import get_embedder

# Ordered high to low, so a group index into this list is a detail level.
_LEVELS_HIGH_TO_LOW = [
    DetailLevel.POINT_CLOUD,
    DetailLevel.BOUNDING_BOX,
    DetailLevel.LABEL,
    DetailLevel.REMOVE,
]

# Cosine cut points, hand set against 0kitchen. A node reaching
# THRESHOLD_POINT_CLOUD is task relevant enough to be manipulated, and the same
# number decides both the object levels here and the seeds of label_radius, so
# there is one number rather than two that can silently disagree.
THRESHOLD_POINT_CLOUD = 0.45
THRESHOLD_BOUNDING_BOX = 0.30
THRESHOLD_LABEL = 0.18

# Cumulative rank fractions for the fixed-quantile cut rule, read off the
# 0kitchen annotations: across its 26 cases the annotators put 6.7 percent of
# the objects at point_cloud, 29.7 at bounding_box, 35.9 at label and the
# remaining 27.7 at remove. Rounded, that is a top 7 percent at point_cloud,
# the next 30 at bounding_box and the next 36 at label. On the 15 objects of
# that scene the rule reproduces the annotators' median composition exactly,
# namely one object at point_cloud, five at bounding_box, five at label and
# four removed. Calibrating on one scene mirrors the practice already used for
# the cosine thresholds and for the information bottleneck's relative floor.
QUANTILE_CUTS = (0.07, 0.37, 0.73)


def is_sub_part(graph: SceneGraph, node_id: str) -> bool:
    """True if the node has a parent object (i.e. it is a functional sub-part)."""
    return bool(graph.get_edges_from_node(node_id))


def parent_id(graph: SceneGraph, node_id: str) -> Optional[str]:
    """Return the id of a sub-part's parent object, or None for a top-level node."""
    edges = graph.get_edges_from_node(node_id)
    return edges[0].target_id if edges else None


def object_ids(node_ids: List[str], graph: SceneGraph) -> List[str]:
    """The top-level objects among ``node_ids``, in input order.

    Sub-parts are excluded because every cut rule in this module works on the
    object distribution alone; sub-parts are filled in afterwards by
    resolve_subparts. This matters most for the quantile rules, where including
    sub-parts would shift the cut points by a scene-dependent amount, the
    FunGraph3D scenes carrying between 4 and 19 objects among 9 to 43 nodes.
    """
    return [nid for nid in node_ids if not is_sub_part(graph, nid)]


def select_seed_ids(node_ids: List[str], scores) -> set:
    """Return ids scoring >= THRESHOLD_POINT_CLOUD, plus the argmax (never empty)."""
    seeds = {nid for nid, s in zip(node_ids, scores) if s >= THRESHOLD_POINT_CLOUD}
    if not seeds and node_ids:
        best = max(range(len(node_ids)), key=lambda i: scores[i])
        seeds.add(node_ids[best])
    return seeds


def bucket(score: float) -> DetailLevel:
    """Map a cosine relevance score to a detail level via fixed thresholds."""
    if score >= THRESHOLD_POINT_CLOUD:
        return DetailLevel.POINT_CLOUD
    if score >= THRESHOLD_BOUNDING_BOX:
        return DetailLevel.BOUNDING_BOX
    if score >= THRESHOLD_LABEL:
        return DetailLevel.LABEL
    return DetailLevel.REMOVE


def assign_objects(node_ids: List[str], graph: SceneGraph,
                   score: Dict[str, float]) -> Dict[str, DetailLevel]:
    """Assign a level to every top-level object from its task-relevance score.

    Objects are bucketed by the fixed thresholds, and the single top-scoring
    object is raised to POINT_CLOUD whatever it scored, so a scene always keeps
    one full-detail anchor even when nothing clears the threshold. Sub-parts are
    skipped here; they are filled afterwards by resolve_subparts.

    The anchor is chosen among objects only, never among sub-parts, so it is
    always something a robot can be said to manipulate. Note that the anchor is
    a floor, not a cap: every other object clearing the threshold reaches
    POINT_CLOUD as well, because a policy is free to hedge between candidates it
    cannot tell apart and the efficiency term prices that hedging.

    This is the shared object-assignment step: the threshold policies differ
    only in the ``score`` they hand in, not in how objects (or sub-parts)
    resolve.
    """
    objects = object_ids(node_ids, graph)
    if not objects:
        return {}
    assignment = {nid: bucket(score[nid]) for nid in objects}
    anchor = max(objects, key=lambda nid: score[nid])
    assignment[anchor] = DetailLevel.POINT_CLOUD
    return assignment


def combined_score(graph: SceneGraph, task: str) -> Dict[str, float]:
    """Relevance per node: node-label similarity or best incident affordance edge.

    A node scores the larger of its own label's similarity to the task and the
    similarity of any affordance edge touching it, so a task can reach an object
    either by naming it or by naming what it affords.

    Shared by combined_inherit and both quantile arms. Keeping the score in one
    place is what makes those three a clean ablation of the cut rule: they are
    literally handed the same numbers and differ only in how the numbers are cut
    into levels.
    """
    node_ids = [n.id for n in graph.nodes]
    embedder = get_embedder()
    sims = embedder.similarity([n.label for n in graph.nodes], [task])[:, 0]
    score = {nid: float(s) for nid, s in zip(node_ids, sims)}

    if graph.edges:
        e_sims = embedder.similarity([e.label for e in graph.edges], [task])[:, 0]
        for edge, s in zip(graph.edges, e_sims):
            s = float(s)
            for nid in (edge.source_id, edge.target_id):
                if nid in score:
                    score[nid] = max(score[nid], s)
    return score


def _levels_from_boundaries(ranked: List[str], boundaries: List[int]
                            ) -> Dict[str, DetailLevel]:
    """Cut a score-ranked id list at the given boundaries into detail levels.

    ``boundaries`` holds ascending rank positions; the group before the first
    boundary is POINT_CLOUD, the next BOUNDING_BOX and so on. Fewer than three
    boundaries therefore leaves the lowest levels unused, filling from the top
    down, which is the deliberate behaviour when a scene offers too few objects
    or too few distinct scores to support four groups. Removing nothing is the
    conservative outcome and is preferred to inventing a cut that the data does
    not support.
    """
    assignment = {}
    group = 0
    cut = list(boundaries)
    for rank, node_id in enumerate(ranked):
        while cut and rank >= cut[0]:
            cut.pop(0)
            group = min(group + 1, len(_LEVELS_HIGH_TO_LOW) - 1)
        assignment[node_id] = _LEVELS_HIGH_TO_LOW[group]
    return assignment


def _ranked_objects(node_ids: List[str], graph: SceneGraph,
                    score: Dict[str, float]) -> List[str]:
    """Objects sorted by score, best first, ties broken by id for reproducibility."""
    return sorted(object_ids(node_ids, graph),
                  key=lambda nid: (-score[nid], nid))


def assign_objects_quantile(node_ids: List[str], graph: SceneGraph,
                            score: Dict[str, float]) -> Dict[str, DetailLevel]:
    """Assign object levels by fixed rank fractions instead of cosine thresholds.

    The objects are ranked by score and cut at QUANTILE_CUTS, so what decides a
    level is an object's position among its peers rather than an absolute
    similarity value. Cosine similarity has no scene-independent meaning, which
    is the standing objection to hand-set thresholds; ranks sidestep it, at the
    price of a different assumption, namely that every scene holds a comparable
    proportion of relevant objects.

    Boundaries are computed half-up and forced to be non-decreasing, and the top
    group is forced to hold at least one object, so the anchor exists even in
    the smallest scenes. On the four objects of 2livingroom this yields one
    object at point_cloud, none at bounding_box, two at label and one removed.

    There is deliberately no absolute floor. A quantile rule promotes its top
    group whatever it scored, so a scene holding nothing relevant to the task
    still keeps an object at full detail, where a threshold rule can correctly
    empty the scene. That is the price of carrying no cosine constant and it is
    reported as a limitation rather than patched with one.
    """
    ranked = _ranked_objects(node_ids, graph, score)
    n = len(ranked)
    if n == 0:
        return {}

    boundaries = []
    previous = 1  # the top group always holds at least the anchor
    for fraction in QUANTILE_CUTS:
        boundary = max(previous, int(fraction * n + 0.5))
        boundaries.append(boundary)
        previous = boundary
    return _levels_from_boundaries(ranked, boundaries)


def assign_objects_adaptive(node_ids: List[str], graph: SceneGraph,
                            score: Dict[str, float]) -> Dict[str, DetailLevel]:
    """Assign object levels by cutting the score ranking at its largest gaps.

    The objects are ranked by score and the three largest drops between
    consecutive scores become the level boundaries. Where the fixed-quantile
    rule assumes a proportion and the threshold rule assumes an absolute value,
    this rule assumes only that a scene separates its relevant objects from its
    irrelevant ones by a visible margin, and it reads that margin off the scene
    at hand. It is the only cut rule here without a hand-set constant.

    Gaps of zero are never chosen, so tied scores cannot be split. A scene
    offering fewer than three positive gaps therefore produces fewer than four
    groups, and the unused levels are the lowest ones (see
    _levels_from_boundaries). Equal gaps are broken towards the higher-scoring
    position so the same input always yields the same cut.

    As with assign_objects_quantile there is no absolute floor; the top group is
    promoted whatever it scored.
    """
    ranked = _ranked_objects(node_ids, graph, score)
    n = len(ranked)
    if n == 0:
        return {}

    gaps = [(score[ranked[i]] - score[ranked[i + 1]], i) for i in range(n - 1)]
    positive = [(gap, i) for gap, i in gaps if gap > 0.0]
    chosen = sorted(positive, key=lambda item: (-item[0], item[1]))[:3]
    boundaries = sorted(i + 1 for _, i in chosen)
    return _levels_from_boundaries(ranked, boundaries)


_SUBPART_FROM_PARENT = {
    DetailLevel.POINT_CLOUD:  DetailLevel.POINT_CLOUD,
    DetailLevel.BOUNDING_BOX: DetailLevel.BOUNDING_BOX,
    DetailLevel.LABEL:        DetailLevel.REMOVE,   # part of a mere landmark = noise
    DetailLevel.REMOVE:       DetailLevel.REMOVE,
}


def resolve_subparts(assignment: Dict[str, DetailLevel], graph: SceneGraph) -> None:
    """Every functional sub-part inherits its parent object's detail level.

    Call after the top-level object levels are assigned. A part of a manipulated
    object is kept with it (PC / BB); a part of a navigation-only (LABEL) or
    removed object is dropped as noise. FunGraph3D is a one-level forest, so a
    sub-part's parent is always a top-level object that already has a level.

    This is the single sub-part rule shared by every inheriting policy, so those
    policies differ only in how they score objects, not in sub-part handling.
    """
    for node in graph.nodes:
        pid = parent_id(graph, node.id)
        if pid is not None:
            assignment[node.id] = _SUBPART_FROM_PARENT[
                assignment.get(pid, DetailLevel.REMOVE)]
