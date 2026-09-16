"""Generic (batch) Agglomerative Information Bottleneck.

Scene-graph-agnostic implementation of Slonim & Tishby's Agglomerative IB
(NIPS 1999), in the form CLIO restates as its Algorithm 1 (Maggio et al.,
2024, Appendix A). The caller supplies priors p(x), task conditionals
p(y|x), an adjacency structure restricting which pairs may ever be merged,
and the mutual-information normaliser of the *whole* problem; this module
repeatedly merges the cheapest adjacent pair until a single merge would
cost more than `delta_bar` of that normaliser.

One formula, two jobs
Slonim & Tishby's proposition 1 states that the merge cost equals the exact
mutual-information decrease caused by that merge:

    d(z_i, z_j) = (p(z_i) + p(z_j)) * JS_PI2[p(y|z_i), p(y|z_j)]        (S&T Fig. 1)
                = I(Z_m;Y) - I(Z_{m-1};Y)                              (S&T prop. 1)

with the *merge prior* PI2 = (p(z_i), p(z_j)) / (p(z_i) + p(z_j)) — a
prior-weighted Jensen-Shannon divergence (S&T Eq. 5), not the uniform one.
The identity holds against any marginal p(y), which cancels:

    p_a*KL(c_a||m) + p_b*KL(c_b||m) - (p_a+p_b)*KL(c_ab||m)
        = (p_a+p_b) * (H(c_ab) - pi_a*H(c_a) - pi_b*H(c_b))
        = (p_a+p_b) * JS_PI2(c_a, c_b)          for every m

so selection and stopping share `_merge_cost` and cannot drift apart.

Stopping criterion
CLIO Eq. 3 defines delta as the loss of a *single* merge relative to the
mutual information of the whole problem, and Algorithm 1 loops
`while delta < delta_bar`; S&T section 3 use the same per-step quantity
delta(m). Run once per connected component of the adjacency graph
(components without edges between them can never produce a cross-component
merge), the per-component form is CLIO Appendix B Eq. 6:

    delta_c(k) = (|X_c| / |X|) * [I((X~_c)_k;Y) - I((X~_c)_{k-1};Y)] / I(X;Y)

With priors on a common scale (CLIO Algorithm 1 line 3 sets p(x_i) = 1/N,
uniform) the two normalising factors collapse and this is simply

    delta_c(k) = merge cost in c / `mi_normalizer`

where `mi_normalizer` is the same number for every component:
`mutual_information_normalizer` over *all* items, using the *global*
marginal. CLIO Appendix B proves this gives the exact same result as
running Algorithm 1 on the full graph. Normalising per component instead —
against that component's own mutual information and its own marginal —
does not: a component carrying little task information would measure every
loss against its own tiny total and stop merging almost immediately.

Deliberate deviations documented in the benchmark formulation:

  - CLIO's Algorithm 1 applies a merge and computes delta afterwards, so
    the merge that breaches the budget is applied. Here the cost is checked
    *before* committing, which makes delta_bar a hard bound. Difference:
    exactly one merge.
  - Only the batch algorithm is implemented: each policy call is a one-shot,
    stateless run over the current graph, so CLIO's incremental Algorithm 2
    (designed for its online mapping session) does not apply here.

Merge costs are recomputed from scratch each iteration rather than updated
only for the pairs touching the new cluster (S&T Fig. 1, last loop step).
That is O(N^2) per step instead of O(N); at the scene-graph sizes this runs
on (tens of nodes) the difference is not measurable.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class IBResult:
    """Final hard cluster assignment from Agglomerative IB."""

    clusters: Dict[str, List[str]]  # cluster_id -> member item ids
    cluster_conditionals: Dict[str, np.ndarray]  # cluster_id -> p(y|x~)
    cluster_priors: Dict[str, float]  # cluster_id -> p(x~)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _weighted_jensen_shannon(
    p: np.ndarray, q: np.ndarray, pi_p: float, pi_q: float
) -> float:
    """JS_PI[p, q] with merge prior PI = (pi_p, pi_q) — S&T Eq. 5.

    The mixture is the prior-weighted one, so it vanishes only where both
    inputs vanish: `_kl_divergence` can never divide by zero here.
    """
    mixture = pi_p * p + pi_q * q
    return pi_p * _kl_divergence(p, mixture) + pi_q * _kl_divergence(q, mixture)


def _merge_cost(
    prior_a: float, prior_b: float, cond_a: np.ndarray, cond_b: np.ndarray
) -> float:
    """d(z_a, z_b) — S&T Fig. 1 / CLIO Eq. 2, exactly the MI decrease."""
    total = prior_a + prior_b
    return total * _weighted_jensen_shannon(
        cond_a, cond_b, prior_a / total, prior_b / total
    )


def mutual_information_normalizer(
    item_ids: List[str],
    priors: Dict[str, float],
    conditionals: Dict[str, np.ndarray],
) -> float:
    """Denominator of CLIO Eq. 6: I(X;Y) scaled by the total prior mass.

    Must be computed over *all* items of the problem (every connected
    component together), because Eq. 6 normalises every component against
    the same global quantity. Priors must be on the same scale as those
    later handed to `run` — a common factor cancels in delta_c(k).
    """
    total_prior = sum(priors[item_id] for item_id in item_ids)
    if total_prior <= 0:
        return 0.0
    marginal = (
        sum(priors[item_id] * conditionals[item_id] for item_id in item_ids)
        / total_prior
    )
    return float(
        sum(
            priors[item_id] * _kl_divergence(conditionals[item_id], marginal)
            for item_id in item_ids
        )
    )


def _find_cheapest_adjacent_pair(
    cluster_priors: Dict[str, float],
    cluster_conditionals: Dict[str, np.ndarray],
    cluster_adjacency: Dict[str, Set[str]],
) -> Optional[Tuple[str, str, float]]:
    """Cheapest mergeable pair and its cost (S&T Fig. 1, `argmin d_ij`).

    Iteration order is sorted so ties resolve to the lexicographically
    smallest pair. S&T allow choosing arbitrarily among several minima;
    fixing the choice keeps runs reproducible.
    """
    best_pair = None
    best_cost = None
    for cid_a in sorted(cluster_adjacency):
        for cid_b in sorted(cluster_adjacency[cid_a]):
            if cid_b <= cid_a:
                continue  # consider each unordered pair once
            cost = _merge_cost(
                cluster_priors[cid_a],
                cluster_priors[cid_b],
                cluster_conditionals[cid_a],
                cluster_conditionals[cid_b],
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pair = (cid_a, cid_b)
    if best_pair is None:
        return None
    return best_pair[0], best_pair[1], best_cost


def run(
    item_ids: List[str],
    priors: Dict[str, float],
    conditionals: Dict[str, np.ndarray],
    adjacency: Dict[str, Set[str]],
    delta_bar: float,
    mi_normalizer: float,
) -> IBResult:
    """Run batch Agglomerative IB on a single connected component.

    Args:
        item_ids: ids of the primitives to cluster.
        priors: p(x) per item id, on the same scale as the priors used for
            `mi_normalizer` (a common factor cancels).
        conditionals: p(y|x) per item id, a probability vector (sums to 1)
            over a shared, fixed set of outcomes y.
        adjacency: item_id -> set of item ids it may be merged with.
        delta_bar: per-merge information-loss budget (CLIO Eq. 3). Merging
            stops before the first merge whose cost, divided by
            `mi_normalizer`, reaches this fraction.
        mi_normalizer: `mutual_information_normalizer` over *all* items of
            the problem, not just this component (CLIO Eq. 6). A value <= 0
            means no task information exists anywhere, so no merge can
            destroy any and merging runs to exhaustion.
    """
    cluster_members: Dict[str, List[str]] = {i: [i] for i in item_ids}
    cluster_priors: Dict[str, float] = {i: priors[i] for i in item_ids}
    cluster_conditionals: Dict[str, np.ndarray] = {
        i: conditionals[i] for i in item_ids
    }
    cluster_adjacency: Dict[str, Set[str]] = {
        i: set(adjacency.get(i, set())) for i in item_ids
    }

    while len(cluster_members) > 1:
        cheapest = _find_cheapest_adjacent_pair(
            cluster_priors, cluster_conditionals, cluster_adjacency
        )
        if cheapest is None:
            break  # no mergeable pairs left in this component

        cid_a, cid_b, cost = cheapest
        if mi_normalizer > 0 and cost / mi_normalizer >= delta_bar:
            break  # CLIO Eq. 3, checked before committing the merge

        new_id = f"{cid_a}+{cid_b}"
        new_prior = cluster_priors[cid_a] + cluster_priors[cid_b]
        new_conditional = (
            cluster_priors[cid_a] * cluster_conditionals[cid_a]
            + cluster_priors[cid_b] * cluster_conditionals[cid_b]
        ) / new_prior
        new_members = cluster_members[cid_a] + cluster_members[cid_b]
        new_neighbors = (
            cluster_adjacency[cid_a] | cluster_adjacency[cid_b]
        ) - {cid_a, cid_b}

        for cid in (cid_a, cid_b):
            del cluster_priors[cid]
            del cluster_conditionals[cid]
            del cluster_members[cid]
            del cluster_adjacency[cid]
        for neighbor in new_neighbors:
            cluster_adjacency[neighbor] = (
                cluster_adjacency[neighbor] - {cid_a, cid_b}
            ) | {new_id}

        cluster_priors[new_id] = new_prior
        cluster_conditionals[new_id] = new_conditional
        cluster_members[new_id] = new_members
        cluster_adjacency[new_id] = new_neighbors

    return IBResult(cluster_members, cluster_conditionals, cluster_priors)
