"""
Ground-truth scoring core for policy results.

Pure computation, no ROS imports and no I/O: compares one policy result
(``node_id -> detail level``) against the ground-truth annotation(s) of
the case and returns the ``gt_scores`` dict that lands in
``evaluation.json`` — written automatically by the evaluator node when a
GT annotation exists for the case, and by ``scripts/score_against_gt.py``
for offline re-scoring (e.g. after calibrating the cost matrix). Cases
whose task target is ambiguous carry several annotations and are scored
best-of-N; file discovery lives in ``gt_files.py``.

Metric: an explicit 4x4 cost matrix over the ordinal DetailLevel scale
``remove < label < bounding_box < point_cloud`` with asymmetric costs
(under-provisioning superlinear, over-provisioning ~ memory cost). Total
cost splits exactly into the lower triangle (under, U) and upper
triangle (over, O), normalised by the worst case achievable on this GT:

    sufficiency S* = 1 - U / U_max      (cost-weighted graded recall)
    efficiency  E* = 1 - O / O_max      (graded specificity)
    q              = 1 - (U + O) / (U_max + O_max)

which makes q a provable convex combination of S* and E*. Matrix values
are placeholders until calibrated against real annotations.

Alongside those scene-wide aggregates the block carries a ``target``
sub-block answering the task-level question — did the policy identify the
one object the task is about and expand exactly it — derived from the
annotation's affordance edges (:func:`build_target_block`).
"""

from collections import defaultdict

from policies.scene_graph import DetailLevel

# Ordinal DetailLevel scale, least to most detail.
LEVELS = [
    DetailLevel.REMOVE.value,
    DetailLevel.LABEL.value,
    DetailLevel.BOUNDING_BOX.value,
    DetailLevel.POINT_CLOUD.value,
]
RANK = {level: i for i, level in enumerate(LEVELS)}

# COST_MATRIX[gt][pred]. Rows must grow monotonically with the distance
# to the diagonal in both directions, otherwise the S*/E*/q
# decomposition does not hold.
COST_MATRIX = {
    "remove":       {"remove": 0.0, "label": 0.1, "bounding_box": 0.3, "point_cloud": 1.0},
    "label":        {"remove": 1.0, "label": 0.0, "bounding_box": 0.2, "point_cloud": 0.6},
    "bounding_box": {"remove": 2.0, "label": 1.5, "bounding_box": 0.0, "point_cloud": 0.3},
    "point_cloud":  {"remove": 4.0, "label": 3.0, "bounding_box": 2.0, "point_cloud": 0.0},
}


def _q_from_costs(under, over, under_max, over_max):
    """Normalised score with the edge case: no possible cost -> 1.0."""
    denominator = under_max + over_max
    if denominator == 0:
        return 1.0
    return 1.0 - (under + over) / denominator


def score_result(gt, pred, excluded_ids=frozenset(), cost_matrix=None):
    """Score one policy result against one ground truth.

    Args:
        gt: ``{node_id: detail_level}`` ground-truth assignment.
        pred: ``{node_id: detail_level}`` policy result assignment.
        excluded_ids: node ids removed from scoring before anything else
            (framework-managed ROOM nodes and routed-condition passage
            objects).
        cost_matrix: optional override of COST_MATRIX (used by the
            sensitivity smoke test).

    Returns a dict with the raw ``gt_scores`` fields (q, sufficiency,
    efficiency, macro_q, under_cost, over_cost, confusion_matrix,
    n_scored, n_excluded) plus ``unmatched_pred_ids`` — nodes present in
    the result but not in the GT. Those are NOT scored (they indicate a
    scene-version mismatch); the caller decides how loudly to warn and
    must not persist that field. Callers that write ``evaluation.json``
    should go through :func:`build_gt_scores_block` instead.

    GT nodes missing from the result count as ``remove`` (a node the
    policy dropped entirely).
    """
    costs = cost_matrix if cost_matrix is not None else COST_MATRIX

    excluded_ids = set(excluded_ids)
    n_excluded = len((set(gt) | set(pred)) & excluded_ids)
    gt = {nid: lvl for nid, lvl in gt.items() if nid not in excluded_ids}
    unmatched_pred_ids = sorted(
        nid for nid in pred if nid not in gt and nid not in excluded_ids
    )

    under = over = under_max = over_max = 0.0
    per_level = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # U, O, U_max, O_max
    confusion = defaultdict(lambda: defaultdict(int))

    for node_id, gt_level in gt.items():
        pred_level = pred.get(node_id, "remove")
        cost = costs[gt_level][pred_level]
        bucket = per_level[gt_level]
        if RANK[pred_level] < RANK[gt_level]:
            under += cost
            bucket[0] += cost
        elif RANK[pred_level] > RANK[gt_level]:
            over += cost
            bucket[1] += cost
        worst_under = costs[gt_level]["remove"]
        worst_over = costs[gt_level]["point_cloud"]
        under_max += worst_under
        over_max += worst_over
        bucket[2] += worst_under
        bucket[3] += worst_over
        confusion[gt_level][pred_level] += 1

    sufficiency = 1.0 - under / under_max if under_max > 0 else 1.0
    efficiency = 1.0 - over / over_max if over_max > 0 else 1.0
    q = _q_from_costs(under, over, under_max, over_max)
    level_scores = [_q_from_costs(*bucket) for bucket in per_level.values()]
    macro_q = sum(level_scores) / len(level_scores) if level_scores else 1.0

    return {
        "q": q,
        "sufficiency": sufficiency,
        "efficiency": efficiency,
        "macro_q": macro_q,
        "under_cost": under,
        "over_cost": over,
        "confusion_matrix": {
            gt_level: dict(row) for gt_level, row in confusion.items()
        },
        "n_scored": len(gt),
        "n_excluded": n_excluded,
        "unmatched_pred_ids": unmatched_pred_ids,
    }


def _point_cloud_parents(levels, edges):
    """point_cloud nodes that are not the source of an affordance edge.

    FunGraph3D edges run from the functional sub-part to the object it
    belongs to ("knob --pull to open or close--> kitchen cabinet"), so a
    node at full detail that is nobody's sub-part is a parent object the
    assignment singled out in its own right.
    """
    sources = {edge["source_id"] for edge in edges}
    return {
        node_id for node_id, level in levels.items()
        if level == DetailLevel.POINT_CLOUD.value and node_id not in sources
    }


def build_target_block(gt_levels, pred_levels, edges):
    """The ``target`` sub-block: did the policy identify the task object?

    The benchmark formulation assumes exactly one target object per task, which together
    with its functional sub-parts deserves full detail. That target is
    recoverable from the annotation alone: it is the one point_cloud node
    that is not a sub-part of anything (see :func:`_point_cloud_parents`).
    The critical node set is the annotation's point_cloud set itself, not
    the target plus every sub-part the edges name — annotators do leave
    individual sub-parts below full detail, and expanding over the edges
    would demand more than the ground truth asks for.

    Fields:
        node_ids: the critical set, so every flag below is auditable.
        parent_id: the target object.
        hit: the whole critical set is at point_cloud in the result. This
            is a feasibility flag — providing *more* than needed keeps it
            true, because the resulting graph still supports the task.
            The waste is what ``efficiency`` measures.
        hit_parent: only the target object itself. Separates "missed the
            object entirely" from "found it but dropped a sub-part".
        n_extra_parents: further parent objects raised to point_cloud,
            i.e. how far the policy over-selected.

    Note: ``excluded_ids`` is deliberately NOT applied here. In the routed
    condition the framework pins passage objects to point_cloud, and
    passages are parents, so ``n_extra_parents`` is inflated there. Rooms
    are unaffected (the framework pins them to LABEL).

    Returns:
        The block, or None when the annotation does not fit the one-target
        assumption (no or several point_cloud parents) so the caller can
        warn instead of persisting a meaningless flag.
    """
    parents = _point_cloud_parents(gt_levels, edges)
    if len(parents) != 1:
        return None
    parent_id = next(iter(parents))

    critical = {
        node_id for node_id, level in gt_levels.items()
        if level == DetailLevel.POINT_CLOUD.value
    }
    pred_point_cloud = {
        node_id for node_id, level in pred_levels.items()
        if level == DetailLevel.POINT_CLOUD.value
    }
    pred_parents = _point_cloud_parents(pred_levels, edges)

    return {
        "node_ids": sorted(critical),
        "parent_id": parent_id,
        "hit": critical <= pred_point_cloud,
        "hit_parent": parent_id in pred_point_cloud,
        "n_extra_parents": len(pred_parents - {parent_id}),
    }


def _score_one(gt_levels, pred_levels, excluded_ids, routing, edges):
    """One result against one annotation, in persisted block shape.

    Scores rounded to 4 decimals, ``feasible`` taken from the routing
    dict and only present in the routed condition (plain condition /
    flat scenes omit the key), ``unmatched_pred_ids`` split off. The
    ``target`` sub-block needs the annotation's affordance edges and is
    omitted without them — a missing key means "not derivable", never a
    failed policy.

    Returns:
        (block, unmatched_pred_ids)
    """
    scores = score_result(gt_levels, pred_levels, excluded_ids=excluded_ids)
    unmatched = scores.pop("unmatched_pred_ids")
    for key in ("q", "sufficiency", "efficiency", "macro_q",
                "under_cost", "over_cost"):
        scores[key] = round(scores[key], 4)
    if routing is not None:
        scores["feasible"] = routing.get("feasible")
    if edges is not None:
        target = build_target_block(gt_levels, pred_levels, edges)
        if target is not None:
            scores["target"] = target
    return scores, unmatched


def build_gt_scores_block(gt_variants, pred_levels, excluded_ids=frozenset(),
                          routing=None, edges=None):
    """Produce the exact ``gt_scores`` block written to evaluation.json.

    Shared by the evaluator node (automatic scoring at run time) and the
    offline CLI so both write byte-identical blocks.

    Args:
        gt_variants: list of ``(variant_id, gt_levels)`` pairs in file
            ground-truth variants. An
            unambiguous case is the single pair ``[(None, gt_levels)]``.
        pred_levels: ``{node_id: detail_level}`` policy result.
        excluded_ids: node ids removed from scoring (ROOM nodes, routed
            passage objects).
        routing: the result's routing block, or None.
        edges: the annotation's affordance edges, needed for the
            ``target`` sub-block (see :func:`build_target_block`). One
            list per case, not per variant: the variants of a case
            annotate the same scene and carry identical edges. Without
            them the sub-block is simply absent.

    Ambiguous cases (a task whose target object exists several times in
    the scene) carry one annotation per candidate target. The task string
    the policy receives contains no disambiguation, so being scored
    against a variant the annotator did not have in mind is not a policy
    error: the result is scored against every variant and the **best**
    one by ``q`` becomes the block, with ties broken on the smallest
    variant id so the choice is reproducible. Those blocks additionally
    carry ``variant_id`` (the winner), ``n_variants``, ``q_mean`` and
    ``per_variant`` with every variant's full block. ``q_mean`` is the
    unweighted mean over the variants, i.e. how much the score depends on
    which target was assumed; it averages the rounded per-variant values
    so a reader can re-derive it from ``per_variant``.

    The variant keys appear whenever the case is variant-annotated, which
    is ``gt_variants[0][0] is not None`` and not simply ``len > 1`` — the
    target id stays worth recording even for a lone variant file. A case
    passed as ``[(None, ...)]`` yields the plain block with none of them.

    Returns:
        (block, unmatched_pred_ids)
    """
    if not gt_variants:
        raise ValueError("build_gt_scores_block needs at least one GT variant")

    scored = [
        (variant_id,)
        + _score_one(gt_levels, pred_levels, excluded_ids, routing, edges)
        for variant_id, gt_levels in gt_variants
    ]

    if scored[0][0] is None:
        _, block, unmatched = scored[0]
        return block, unmatched

    # Best of N by q; the id breaks ties so re-scoring the same inputs
    # always names the same winner.
    ranked = sorted(scored, key=lambda item: (-item[1]["q"], item[0]))
    best_id, best_block, _ = ranked[0]

    block = dict(best_block)
    block["variant_id"] = best_id
    block["n_variants"] = len(scored)
    block["q_mean"] = round(
        sum(item[1]["q"] for item in scored) / len(scored), 4
    )
    block["per_variant"] = {
        variant_id: variant_block for variant_id, variant_block, _ in scored
    }

    unmatched = sorted({
        node_id for _, _, variant_unmatched in scored
        for node_id in variant_unmatched
    })
    return block, unmatched
