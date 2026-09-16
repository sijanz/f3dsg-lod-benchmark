"""Task-driven Agglomerative IB over functional scene graphs — shared core.

Runner behind `information_bottleneck` and `information_bottleneck_noaff`;
the two policy modules differ only in the `affordances` flag.

The algorithm is Slonim & Tishby's Agglomerative IB (`utils/agglomerative_ib`),
in the form CLIO restates as its Algorithm 1 (Maggio et al., 2024). This module
holds the *modelling* layer: how p(y|x), the priors and the merge adjacency are
derived from a functional scene graph, and how the resulting clusters become
detail levels. Where CLIO's own choices carry over they are used unchanged;
each deviation is named below with the reason it is necessary here.

Carried over from CLIO unchanged
  - Uniform priors p(x_i) = 1/N (Algorithm 1, line 3).
  - The null task: a primitive whose best task similarity falls below a floor
    gets the one-hot null conditional [1, 0] and can never be task-relevant
    (Section V-B, "pre-pruning step").
  - The stopping criterion delta_bar (Eq. 3) and its per-component form
    (Eq. 6), both implemented in `utils/agglomerative_ib`.
  - Merging is restricted to spatially adjacent primitives.

Deviations, and why
  - **Single task.** CLIO ranks m tasks, so its Y has m+1 outcomes. Here Y is
    binary, {null, task}. CLIO's Eq. 5 specialised to m = 1 reads
    p(y|x) = [alpha/(alpha+phi), phi/(alpha+phi)], which is exactly this
    module's mapping at GAMMA = 1 with an absolute floor — so the faithful
    configuration is reachable and GAMMA is a single, named extension of it
    rather than an unrelated formula. A binary Y also means p(y|x) is fixed by
    one scalar; the resulting degeneracy is discussed in the benchmark formulation.
  - **Affordance-edge-aware relevance.** A node's task similarity is the best
    match of its own label OR any incident affordance phrase combined with the
    neighbouring node's label ("pull to open or close the microwave oven").
    Edge labels are the core asset of functional scene graphs; label-only
    matching cannot see that a plain "handle" belongs to the microwave. The
    edge label alone is ambiguous (the same phrase hangs on every cabinet
    knob), so it is only ever scored together with the other endpoint's label.
  - **Text-text instead of image-text similarity.** CLIO measures CLIP
    image-text cosines and sets alpha = 0.23 (Section V-B) — raised to 0.26 for
    OpenCLIP ViT-H-14 (Appendix H), which shows the constant is calibrated per
    embedding model rather than universal. Here both sides are text, whose
    similarity carries a much higher generic baseline (any kitchen furniture
    matches any kitchen task somewhat), so FLOOR_MODE allows a scene-relative
    floor as an alternative to the absolute one.
  - **Spatial-distance adjacency.** CLIO adds an edge between primitives whose
    3D bounding boxes have non-zero overlap (Section V-B). The minimal graph a
    policy receives has no `obb`, so proximity of the node positions stands in.
  - **The cluster -> detail-level mapping has no CLIO counterpart.** CLIO's
    output is binary (keep a primitive or drop it), so it needs no such step;
    alpha is its null-task score, not a level threshold. Turning a cluster's
    task probability into one of four detail levels is this benchmark formulation's own
    construction and is deliberately kept separate from the IB step above.
  - **The hierarchy filter.** CLIO's primitives are over-segmented fragments, so
    merging *reconstructs* objects. Here the primitives are already whole,
    named objects carrying affordance edges, so merging them would destroy
    exactly the structure a robot needs to interact. Clusters at the
    manipulation level therefore share a detail level but keep their identity;
    only coarser clusters are collapsed into one node.
"""

from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from policies.utils.agglomerative_ib import (
    mutual_information_normalizer,
    run as run_agglomerative_ib,
)
from policies.utils.text_embedder import get_embedder
from policies.scene_graph import (
    CONTAINS_EDGE_LABEL,
    DetailLevel,
    NEUTRAL_RELATION_LABEL,
    Node,
    PASSAGE_EDGE_LABEL,
    PolicyResult,
    REACHABLE_EDGE_LABEL,
    SceneGraph,
)


# Absolute lower bound for task relatedness — CLIO's alpha (Section V-B,
# "We set alpha = 0.23"). Below this raw similarity a node never counts as
# relevant, which also guards the "nothing in the scene relates to the task"
# case.
NULL_SIMILARITY_FLOOR = 0.23

# "absolute" uses NULL_SIMILARITY_FLOOR as-is (CLIO's formulation), "relative"
# raises the floor to a fraction of the scene's best score.
#
# Calibrated on 0kitchen over the four calibration tasks; the faithful
# absolute mode was measured first and is inadequate here for a structural
# reason: at alpha = 0.23 only 3 of the scene's 39 nodes fall below the floor,
# so at most 3 nodes can ever reach REMOVE, while the ground-truth annotations
# drop 14 on average. Text-text similarity simply has no comparable null mass
# to CLIP's image-text similarity, which is the same effect that makes CLIO
# retune alpha per embedding model (0.23 -> 0.26 in its Appendix H).
FLOOR_MODE = "relative"
RELATIVE_FLOOR_FRACTION = 0.5

# Contrast exponent for the task conditionals:
#   p_task = phi^GAMMA / (floor^GAMMA + phi^GAMMA)
# GAMMA = 1 is exactly CLIO Eq. 5 for a single task; larger values sharpen the
# contrast between strongly and mildly relevant nodes so those distinctions
# carry more mutual information and survive the bottleneck.
#
# Also calibrated against the faithful setting: at GAMMA = 1 the near-field
# band of generic parts ("pull to open or close the kitchen cabinet") is not
# separated from the task object at all — IB returns one 21-node cluster
# holding both. At GAMMA = 3 the same scene splits into task object + parts
# (8), context band (13) and background (18), which is the separation the
# detail levels are meant to express.
GAMMA = 3.0

# Edges with structural (non-affordance) meaning never contribute to task
# relevance.
_STRUCTURAL_EDGE_LABELS = {
    CONTAINS_EDGE_LABEL,
    REACHABLE_EDGE_LABEL,
    PASSAGE_EDGE_LABEL,
}

# Stand-in for CLIO's non-zero bounding-box overlap (graph units ~ metres).
ADJACENCY_DISTANCE_THRESHOLD = 1.5

# CLIO Eq. 3: stop before the first merge that would cost more than this
# fraction of the scene's total mutual information with the task. Calibrated
# on 0kitchen: the clustering is identical for every value from 0.01 to 0.08,
# so this sits in the middle of a wide plateau rather than on a knife edge.
# At 0.09 the task object and the context band collapse into one cluster.
DELTA_BAR = 0.05

# Detail-level cutoffs on the *normalised* cluster score (see
# `_score_to_detail_level`). Not inherited from CLIO, which has no such step:
# its alpha is a null-task score, not a level threshold.
#
# T_BBOX is the one principled value: a cluster sitting exactly on the null
# floor scores 0.5 raw, i.e. 0.5 / score_max = 0.56 normalised, and under a
# relative floor score_max = 1/(1 + fraction^GAMMA) is a scene-invariant
# constant, so 0.56 always marks "still genuinely relevant" against "a mixture
# diluted by null members". T_PC sits midway between the weakest task cluster
# measured over the four calibration tasks (0.850, the sink) and the strongest
# context band (0.614, the generic cabinet handles and knobs). T_LABEL is half
# the floor point; no calibration case produced a cluster in that band.
T_LABEL = 0.28
T_BBOX = 0.56
T_PC = 0.75


def _node_task_similarities(
    graph: SceneGraph, task: str, affordances: bool = True
) -> Dict[str, float]:
    """phi per node: best task similarity over its candidate texts.

    Candidates are the node's own label plus, per incident affordance edge,
    the combined phrase "<edge label> the <other endpoint's label>". With
    `affordances=False` the edge label is replaced by the neutral relation
    while the neighbour's label is kept, so the withheld condition removes the
    affordance text alone and not the neighbour's identity along with it. One
    embedder pass over the deduplicated candidate texts.
    """
    node_by_id = {n.id: n for n in graph.nodes}
    candidate_texts: Dict[str, List[str]] = {
        n.id: [n.label] for n in graph.nodes
    }
    for edge in graph.edges:
        if edge.label in _STRUCTURAL_EDGE_LABELS:
            continue
        source = node_by_id.get(edge.source_id)
        target = node_by_id.get(edge.target_id)
        if source is None or target is None:
            continue
        relation = edge.label if affordances else NEUTRAL_RELATION_LABEL
        candidate_texts[source.id].append(f"{relation} the {target.label}")
        candidate_texts[target.id].append(f"{relation} the {source.label}")

    unique_texts = sorted(
        {text for texts in candidate_texts.values() for text in texts}
    )
    embedder = get_embedder()
    raw = embedder.similarity(unique_texts, [task])[:, 0]
    sim_by_text = {text: float(s) for text, s in zip(unique_texts, raw)}
    return {
        node_id: max(sim_by_text[text] for text in texts)
        for node_id, texts in candidate_texts.items()
    }


def _task_floor(similarities: Dict[str, float]) -> float:
    """The null-task floor, absolute (CLIO's alpha) or scene-relative."""
    if FLOOR_MODE == "relative":
        return max(
            NULL_SIMILARITY_FLOOR,
            RELATIVE_FLOOR_FRACTION * max(similarities.values()),
        )
    return NULL_SIMILARITY_FLOOR


def _build_conditionals(
    node_ids: List[str], similarities: Dict[str, float], floor: float
) -> Dict[str, np.ndarray]:
    """p(y|x) as [p_null, p_task] per node — CLIO Eq. 5 for a single task."""
    conditionals = {}
    floor_g = floor ** GAMMA
    for node_id in node_ids:
        phi = similarities[node_id]
        if phi < floor:
            conditionals[node_id] = np.array([1.0, 0.0])
        else:
            phi_g = phi ** GAMMA
            conditionals[node_id] = np.array([floor_g / (floor_g + phi_g),
                                              phi_g / (floor_g + phi_g)])
    return conditionals


def _build_adjacency(nodes: List[Node]) -> Dict[str, Set[str]]:
    """Merge-candidate graph — proximity stands in for CLIO's bbox overlap."""
    adjacency: Dict[str, Set[str]] = {n.id: set() for n in nodes}
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1:]:
            distance = float(
                np.linalg.norm(
                    np.array(node_a.position) - np.array(node_b.position)
                )
            )
            if distance <= ADJACENCY_DISTANCE_THRESHOLD:
                adjacency[node_a.id].add(node_b.id)
                adjacency[node_b.id].add(node_a.id)
    return adjacency


def _connected_components(
    node_ids: List[str], adjacency: Dict[str, Set[str]]
) -> List[List[str]]:
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for node_id, neighbors in adjacency.items():
        for neighbor in neighbors:
            graph.add_edge(node_id, neighbor)
    return [sorted(component) for component in nx.connected_components(graph)]


def _score_to_detail_level(score_norm: float) -> DetailLevel:
    """Map a cluster's normalised task probability to a detail level.

    The raw cluster score is p_task of the cluster conditional, a
    prior-weighted mean over its members, so it lies in [0, score_max] with
    score_max = phi_max^GAMMA / (floor^GAMMA + phi_max^GAMMA). Under a
    relative floor score_max is a scene-invariant constant, under an absolute
    floor it is not; dividing by it puts both floor modes on the same [0, 1]
    scale so one set of cutoffs stays meaningful for either. A cluster sitting
    exactly on the floor scores 0.5 raw, and pure null clusters score 0.
    """
    if score_norm < T_LABEL:
        return DetailLevel.REMOVE
    if score_norm < T_BBOX:
        return DetailLevel.LABEL
    if score_norm < T_PC:
        return DetailLevel.BOUNDING_BOX
    return DetailLevel.POINT_CLOUD


def apply_ib(
    graph: SceneGraph, task: str, policy_name: str, affordances: bool = True
) -> PolicyResult:
    """Cluster task-similar, spatially adjacent nodes via Agglomerative IB.

    Runs CLIO's Algorithm 1 once per connected component of the proximity
    graph, with the stopping criterion normalised against the whole scene
    (Eq. 6). Clusters then receive a detail level; clusters below the
    manipulation level and holding at least two nodes are additionally
    collapsed into one node via `merge_assignment`, while clusters at
    POINT_CLOUD keep their members as individual nodes so functional parts
    and their affordance edges survive into the product graph.
    """
    node_ids = sorted(node.id for node in graph.nodes)
    detail_assignment: Dict[str, DetailLevel] = {}
    merge_assignment: Dict[str, str] = {}

    def result(metadata=None):
        return PolicyResult(
            policy_name=policy_name,
            task=task,
            detail_assignment=detail_assignment,
            merge_assignment=merge_assignment,
            metadata=metadata or {},
        )

    if not node_ids:
        return result()

    similarities = _node_task_similarities(graph, task, affordances)
    floor = _task_floor(similarities)
    phi_max = max(similarities.values())

    # If nothing in the scene passes the floor, the task relates to no node
    # at all and the whole graph collapses.
    if phi_max < floor:
        for node_id in node_ids:
            detail_assignment[node_id] = DetailLevel.REMOVE
        return result({"ib": {
            "floor": floor, "floor_mode": FLOOR_MODE, "gamma": GAMMA,
            "phi_max": phi_max, "task_related": False, "clusters": [],
        }})

    conditionals = _build_conditionals(node_ids, similarities, floor)
    adjacency = _build_adjacency(graph.nodes)
    # CLIO Algorithm 1, line 3: p(x_i) = 1/N. Passing 1.0 is the same
    # distribution up to a common factor, which cancels in delta_c(k).
    priors = {node_id: 1.0 for node_id in node_ids}

    # CLIO Eq. 6 normalises every component against the mutual information of
    # the whole scene, computed with the global marginal — not against each
    # component's own, which would let information-poor components stop
    # merging almost immediately.
    mi_normalizer = mutual_information_normalizer(
        node_ids, priors, conditionals
    )

    components = _connected_components(node_ids, adjacency)
    clusters: List[Tuple[List[str], float]] = []
    for component_ids in components:
        component_adjacency = {
            node_id: adjacency[node_id] & set(component_ids)
            for node_id in component_ids
        }
        ib_result = run_agglomerative_ib(
            item_ids=component_ids,
            priors=priors,
            conditionals=conditionals,
            adjacency=component_adjacency,
            delta_bar=DELTA_BAR,
            mi_normalizer=mi_normalizer,
        )
        for cluster_id, members in ib_result.clusters.items():
            score = float(ib_result.cluster_conditionals[cluster_id][1])
            clusters.append((sorted(members), score))

    score_max = phi_max ** GAMMA / (floor ** GAMMA + phi_max ** GAMMA)
    # Most task-relevant cluster first; member id breaks ties deterministically.
    clusters.sort(key=lambda item: (-item[1], item[0][0]))

    cluster_records = []
    for index, (members, score) in enumerate(clusters):
        score_norm = score / score_max
        level = _score_to_detail_level(score_norm)
        group_id = f"ib_cluster_{index}"

        for member_id in members:
            detail_assignment[member_id] = level
        # Hierarchy filter: only clusters below the manipulation level are realised
        # as a single merged node. At POINT_CLOUD the members stay individual
        # nodes, which is what keeps their affordance edges alive — the graph
        # modifier only drops an edge whose endpoints land in the same merge
        # group, or whose endpoint is removed.
        merged = level in (DetailLevel.BOUNDING_BOX, DetailLevel.LABEL) and (
            len(members) > 1
        )
        if merged:
            for member_id in members:
                merge_assignment[member_id] = group_id

        cluster_records.append({
            "id": group_id,
            "members": members,
            "size": len(members),
            "score": score,
            "score_norm": score_norm,
            "level": level.value,
            "merged": merged,
        })

    # Everything an offline re-mapping needs: the clustering does not depend
    # on the level mapping (it is computed before any threshold is applied),
    # so an alternative mapping — for instance CLIO's binary keep/drop — can
    # be derived from these records exactly, without re-running the pipeline.
    return result({"ib": {
        "floor": floor,
        "floor_mode": FLOOR_MODE,
        "gamma": GAMMA,
        "delta_bar": DELTA_BAR,
        "stop_rule": "per_step",
        "affordances": affordances,
        "phi_max": phi_max,
        "score_max": score_max,
        "global_mi": mi_normalizer,
        "n_components": len(components),
        "thresholds": {"label": T_LABEL, "bbox": T_BBOX, "pc": T_PC},
        "task_related": True,
        "clusters": cluster_records,
    }})
