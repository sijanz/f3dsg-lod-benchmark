# Benchmarking Task-Driven Level-of-Detail Abstraction for Functional 3D Scene Graphs

This repository contains the official benchmark release and evaluation artifacts accompanying the paper:
> **"Benchmarking Task-Driven Level-of-Detail Abstraction for Functional 3D Scene Graphs"**

The benchmark evaluates how to selectively abstract 3D scene graphs into tiered geometric representations (`point_cloud`, `bounding_box`, `label`, `remove`) for robot manipulation tasks. It provides:
1. **A Kinematic Ground-Truth Derivation Reference:** Derived from robot reachability and base placement kinematics over 232 mobile manipulation tasks across 14 indoor scenes.
2. **A Cost-Weighted Balanced Resolution Metric ($Q$):** Formalizes sufficiency ($S^*$) and efficiency ($E^*$) with derived closed-form weights ($w_S = 0.549414, w_E = 0.450586$).
3. **Ten Evaluated Abstraction Policies:** Implementations and frozen hyperparameter signatures for geometric, semantic, inheritance, and information-theoretic policies.
4. **Planning Benchmark Evaluation:** Full batch execution records and aggregate statistics over 3,720 MoveIt 2 motion planning trials (155 placed cases $\times$ 8 policy arms $\times$ 3 repeats).
5. **Physical Hardware Replay Study:** Joint trajectories and empirical execution records across 3 manipulation cases on an AgileX Piper 6-DoF robotic arm.

---

## Table of Contents

- [Benchmarking Task-Driven Level-of-Detail Abstraction for Functional 3D Scene Graphs](#benchmarking-task-driven-level-of-detail-abstraction-for-functional-3d-scene-graphs)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Representation Levels \& Formulation](#representation-levels--formulation)
    - [The Metric ($Q$)](#the-metric-q)
  - [Frozen Kinematic Constants](#frozen-kinematic-constants)
  - [The Ten Evaluated Policies](#the-ten-evaluated-policies)
  - [Benchmark Results Table](#benchmark-results-table)
    - [Key Analytical Findings:](#key-analytical-findings)
  - [Quickstart: Reproducing Results](#quickstart-reproducing-results)
    - [1. Environment Setup](#1-environment-setup)
    - [2. Regenerate Main Results Table](#2-regenerate-main-results-table)
    - [3. Derive Metric Weights \& Invariants](#3-derive-metric-weights--invariants)
    - [4. Reproduce Planning Benchmark Aggregates](#4-reproduce-planning-benchmark-aggregates)
    - [5. Run Kinematic Sensitivity Sweep](#5-run-kinematic-sensitivity-sweep)
    - [6. Trajectory Replay Execution](#6-trajectory-replay-execution)
  - [Release Notes](#release-notes)

---

## Repository Structure

```
.
├── reference/
│   └── derive_ground_truth.py        # Ground-truth kinematic reference & placement solver
├── policies/
│   ├── identity_policy.py            # Baseline 1: Identity (all nodes at point_cloud)
│   ├── label_radius_policy.py        # Baseline 2: Radial distance shells around seeds
│   ├── combined_quantile_adaptive_policy.py # Policy 3: Adaptive quantile cuts (largest gap)
│   ├── combined_quantile_policy.py   # Policy 4: Fixed quantile cuts (7% / 30% / 36%)
│   ├── combined_inherit_policy.py    # Policy 5: Combined score graph edge inheritance
│   ├── label_inherit_policy.py       # Policy 6: Label score graph edge inheritance
│   ├── label_only_policy.py          # Policy 7: Semantic label cosine thresholding
│   ├── affordance_inherit_policy.py  # Policy 8: Affordance inheritance with fallback
│   ├── information_bottleneck_policy.py # Policy 9: Information bottleneck with affordances
│   ├── information_bottleneck_noaff_policy.py # Policy 10: Information bottleneck without affordances
│   ├── policies_registry.py          # Canonical registry for the 10 paper policies
│   ├── scene_graph.py                # Scene graph data structures (nodes, edges, detail levels)
│   ├── dataset_adapters/             # FunGraph3D scene graph dataset loader & adapters
│   └── utils/                        # Scoring, text embedder, clustering, and graph filtering
├── metric/
│   ├── gt_scoring.py                 # Core metric computation (Sufficiency S*, Efficiency E*, Q)
│   └── generate_metric_weights.py    # Closed-form weight derivation (w_S = 0.549414, w_E = 0.450586)
├── evaluation/
│   ├── all_10_arms_evaluation.json   # Canonical evaluation artifact across 10 policies
│   ├── report_results.py             # Leaderboard, level distributions, and invariant verifier
│   ├── generate_evaluation_artifact.py # Evaluation harness generating evaluation artifact
│   └── run_sensitivity_sweep.py      # Kinematic parameter sensitivity sweep runner
├── planning/
│   ├── run_planning_benchmark.py     # Batch runner for 3,720 MoveIt 2 planning trials
│   ├── aggregate_planning_results.py # Aggregator computing success rates & collision statistics
│   ├── planning_benchmark_results.jsonl # Complete raw execution records for all 3,720 trials
│   ├── support/                      # MoveIt 2 scene builder, planner, KDL kinematics, OctoMap bridge
│   └── launch/                       # Headless MoveIt 2 simulation launch file
├── replay/
│   ├── replay_executor.py            # ROS 2 FollowJointTrajectory action client for trajectory replay
│   └── bundles/                      # Frozen trajectory execution bundles for 3 evaluation cases
├── data/
│   ├── tasks_target_objects.json     # 232 benchmark task specifications (JSON, SHA-256: 28ee7be9...)
│   ├── tasks_target_objects.csv      # 232 benchmark task specifications (CSV, SHA-256: 4cef88b3...)
│   ├── all_232_tasks_evaluation.json # Placement solver reference outcomes (155 placed, 77 unplaced)
│   ├── ground_truth_kinematics.json  # Kinematic targets and base poses for 155 placed cases
│   ├── ground_truth_kinematics_reselected.json # Margin-maximized target configurations
│   ├── node_geom.json                # Precomputed 3D bounding boxes and centroids for all scene nodes
│   └── affordance_heights.json       # Calibrated scene floor elevations
├── scripts/
│   ├── reproduce_table.sh            # One-step command: Regenerate main paper results table
│   ├── reproduce_metric_weights.sh   # One-step command: Derive metric weights and verify invariants
│   ├── reproduce_planning_aggregate.sh # One-step command: Re-aggregate 3,720 planning trial results
│   └── reproduce_sensitivity.sh      # One-step command: Execute parameter sensitivity sweep
├── requirements.txt                  # Python dependencies
└── README.md                         # Benchmark documentation
```

---

## Representation Levels & Formulation

Every node in the scene graph is assigned exactly one detail level:
- **`point_cloud` (Level 3):** Full high-resolution geometric detail. Required for manipulation target objects and their functional mechanical affordances.
- **`bounding_box` (Level 2):** 3D Oriented Bounding Box (OBB). Sufficient for collision avoidance of proximal obstacles along the robot's base/arm reach envelope.
- **`label` (Level 1):** Semantic label and 3D centroid position only (0 collision geometry). Retains semantic spatial awareness with minimal data overhead.
- **`remove` (Level 0):** Completely pruned from the representation. Distant or irrelevant nodes that do not affect navigation or manipulation.

### The Metric ($Q$)
The metric penalizes under-provisioning ($U$, assigning less detail than ground truth) and over-provisioning ($O$, allocating excess detail):
$$\begin{aligned}
S^* &= 1 - \frac{U}{U_{\max}} \quad \text{(Sufficiency / Graded Recall)} \\
E^* &= 1 - \frac{O}{O_{\max}} \quad \text{(Efficiency / Graded Specificity)} \\
Q &= 1 - \frac{U + O}{U_{\max} + O_{\max}} = w_S S^* + w_E E^*
\end{aligned}$$
where:
- $w_S = \frac{U_{\max}}{U_{\max} + O_{\max}} = \frac{4001.0}{7282.3} \approx 0.549414$
- $w_E = \frac{O_{\max}}{U_{\max} + O_{\max}} = \frac{3281.3}{7282.3} \approx 0.450586$

---

## Frozen Kinematic Constants

All reference placement calculations and ground-truth derivations use frozen physical constants defined in `reference/derive_ground_truth.py`:

| Constant Parameter | Frozen Value | Physical Meaning |
| :--- | :---: | :--- |
| `BASE_FOOTPRINT_WIDTH` | `0.30 m` | Width of the square mobile robot base footprint |
| `ARM_MIN_REACH` | `0.034 m` | Inner kinematic radius limit of the manipulator arm |
| `ARM_MAX_REACH` | `0.626 m` | Outer spherical reach radius of the manipulator arm |
| `REACH_MARGIN` | `0.05 m` | Margin added to nominal reach radius |
| `COLLISION_MARGIN` | `0.05 m` | Obstacle inflation distance for safe manipulator clearance |
| `BASE_CLEARANCE` | `0.02 m` | Stand-off clearance from obstacles to mobile base footprint |
| `GRID_RESOLUTION` | `0.02 m` | Spatial discretization resolution for base placement raster grid |
| `MOUNT_HEIGHT_MIN` | `0.30 m` | Lower bound of the torso mount height search range |
| `MOUNT_HEIGHT_MAX` | `1.40 m` | Upper bound of the torso mount height search range |
| `R_NOMINAL_FRACTION` | `0.75` | Fraction of max reach for preferred stance ($0.75 \times 0.626 = 0.4695\text{ m}$) |

Across the 232 benchmark tasks, the solver determines:
- **155 Placed Tasks:** A feasible base stance and valid inverse kinematics solution exist.
- **77 Unplaced Tasks:** No collision-free base stance can reach the affordance (4 vertically infeasible, 73 footprint blocked by surrounding scene geometry).

---

## The Ten Evaluated Policies

All policies are executed with identical frozen parameter signatures:

| # | Policy Name | Strategy | Key Parameters & Thresholds |
| :-: | :--- | :--- | :--- |
| **1** | `identity` | Full detail baseline | All nodes $\to$ `point_cloud` (0 parameters) |
| **2** | `label_radius` | Euclidean spatial shells | $r_{\text{pc}} \le 0.75\text{m}$, $r_{\text{bb}} \le 1.50\text{m}$, $r_{\text{lbl}} \le 2.50\text{m}$, seed threshold $\ge 0.45$ |
| **3** | `combined_quantile_adaptive` | Adaptive ranking gap | Splits ranked nodes by maximum score gap (0 parameters) |
| **4** | `combined_quantile` | Fixed rank cuts | Quantile boundaries: $7\% \to \text{pc}$, $30\% \to \text{bb}$, $36\% \to \text{label}$ |
| **5** | `combined_inherit` | Graph edge inheritance | Cosine thresholds: $0.45 / 0.30 / 0.18$ over combined label + affordance scores |
| **6** | `label_inherit` | Graph edge inheritance | Cosine thresholds: $0.45 / 0.30 / 0.18$ over label score; inherits across edges |
| **7** | `label_only` | Structure-blind semantic | Cosine thresholds: $0.45 / 0.30 / 0.18$ over label cosine similarity only |
| **8** | `affordance_inherit` | Functional inheritance | Matches affordances; fallback to label with discount factor $\times 0.8$ |
| **9** | `information_bottleneck` | Information-theoretic | Agglomerative IB with affordances ($\gamma = 3.0$, floor fraction $0.50$, $\Delta = 0.05$) |
| **10** | `information_bottleneck_noaff` | Information-theoretic | Agglomerative IB without affordances ($\gamma = 3.0$, floor fraction $0.50$, $\Delta = 0.05$) |

*Text Embedder:* All embedding-based policies use the SentenceTransformer model `all-MiniLM-L6-v2`.

---

## Benchmark Results Table

Regenerating `scripts/reproduce_table.sh` produces the exact leaderboard and metrics reported in the paper:

```
Arm / Policy                 | q (pool) | q (macro) | S*       | E*       | q_excl   | S*_excl  | E*_excl  | Target%  | Extra  | Dense  | Size  
---------------------------------------------------------------------------------------------------------------------------------------
identity                     | 0.5494   | 0.5679    | 1.0000   | 0.0000   | 0.3944   | 1.0000   | 0.0000   | 100.0    | 10.05  | 1.00   | 1.00  
label_radius                 | 0.7837   | 0.8018    | 0.8980   | 0.6443   | 0.7545   | 0.9237   | 0.6443   | 74.2     | 2.41   | 0.35   | 0.31  
combined_quantile_adaptive   | 0.7334   | 0.7560    | 0.6997   | 0.7746   | 0.7060   | 0.6006   | 0.7746   | 76.8     | 1.05   | 0.22   | 0.22  
combined_quantile            | 0.7208   | 0.7475    | 0.6123   | 0.8530   | 0.7222   | 0.5213   | 0.8530   | 63.9     | 0.36   | 0.13   | 0.14  
combined_inherit             | 0.7086   | 0.7415    | 0.6626   | 0.7648   | 0.6772   | 0.5428   | 0.7648   | 78.1     | 1.50   | 0.26   | 0.26  
label_inherit                | 0.7105   | 0.7412    | 0.6238   | 0.8161   | 0.6823   | 0.4768   | 0.8161   | 76.8     | 1.23   | 0.23   | 0.23  
label_only                   | 0.6784   | 0.6656    | 0.5187   | 0.8731   | 0.7548   | 0.5730   | 0.8731   | 8.4      | 1.23   | 0.08   | 0.23  
affordance_inherit           | 0.5872   | 0.6038    | 0.3834   | 0.8358   | 0.6562   | 0.3804   | 0.8358   | 23.9     | 0.99   | 0.14   | 0.14  
information_bottleneck       | 0.6460   | 0.7001    | 0.6018   | 0.6998   | 0.5910   | 0.4240   | 0.6998   | 65.8     | 2.40   | 0.35   | 0.37  
information_bottleneck_noaff | 0.6579   | 0.7107    | 0.5991   | 0.7296   | 0.5827   | 0.3570   | 0.7296   | 79.4     | 1.96   | 0.35   | 0.34  
```

### Key Analytical Findings:
- **Affordance Gap:** `combined_quantile_adaptive` ($Q = 0.7334$, Target Hit $= 76.8\%$) outperforms `affordance_inherit` ($Q = 0.5872$, Target Hit $= 23.9\%$) by $\Delta Q = +0.1462$ and $+52.9\%$ target retention.
- **Hierarchy Propagation:** `label_inherit` ($Q = 0.7105$, Target Hit $= 76.8\%$) recovers structured context over `label_only` ($Q = 0.6784$, Target Hit $= 8.4\%$), representing a $+68.4\%$ boost in target hit rate.

---

## Quickstart: Reproducing Results

### 1. Environment Setup

Clone and install dependencies:
```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
```

### 2. Regenerate Main Results Table
Regenerates the complete 10-arm leaderboard, level distributions, and invariant checks from the released evaluation artifact:
```bash
bash scripts/reproduce_table.sh
```

### 3. Derive Metric Weights & Invariants
Computes the penalty sums $U_{\max} = 4001.0$ and $O_{\max} = 3281.3$ and verifies $w_S = 0.549414, w_E = 0.450586$:
```bash
bash scripts/reproduce_metric_weights.sh
```

### 4. Reproduce Planning Benchmark Aggregates
Recomputes statistical tables, Wilson score intervals, and collision statistics over the 3,720 MoveIt 2 trials:
```bash
bash scripts/reproduce_planning_aggregate.sh
```
This script parses the 3,720 execution records from `planning/planning_benchmark_results.jsonl` and computes aggregate statistics, success rates, and collision frequencies.

### 5. Run Kinematic Sensitivity Sweep
Executes the parameter sweep across collision margins ($0.00\text{--}0.20\text{ m}$), mount heights, support tolerances, and nominal reach radii:
```bash
bash scripts/reproduce_sensitivity.sh
```

### 6. Trajectory Replay Execution
The physical replay runner communicates with a standard ROS 2 `FollowJointTrajectory` action server (`/arm_controller/follow_joint_trajectory`):
```bash
# Execute replay of Case 1 (10kitchen open cabinet GT trajectory)
python3 replay/replay_executor.py --bundle replay/bundles/bundle_10kitchen_open-the-kitchen-cabinet_afb1db89_GT.json

# Execute replay of Case 2 (10kitchen open oven GT trajectory)
python3 replay/replay_executor.py --bundle replay/bundles/bundle_10kitchen_open-the-oven_GT.json

# Execute replay of Case 3 (13bathroom trash GT trajectory)
python3 replay/replay_executor.py --bundle replay/bundles/bundle_13bathroom_i-need-to-throw-away-trash_GT.json
```
Empirical joint trajectories and execution bundles for all 3 physical trials are provided in `replay/bundles/`.

---

## Release Notes

1. **FunGraph3D Scene Mesh Data:**
   All benchmark evaluations run offline using precomputed geometry in `data/node_geom.json` and task definitions in `data/tasks_target_objects.json`. For users wishing to re-derive base placements from raw 3D mesh scans, raw ScanNet/FunGraph3D `.ply` point cloud files should be downloaded into `dataset/FunGraph3D/<scene_id>/<scene_id>.ply`.
2. **Anonymization Notice:**
   In compliance with double-blind review policies, all author identities, institutional affiliations, and internal machine paths have been scrubbed from this release.