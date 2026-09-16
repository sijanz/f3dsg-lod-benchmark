"""
FunGraph3D Dataset Loader.

Loads scene graphs from the FunGraph3D dataset directory.
Constructs SceneGraph objects with positions computed from pointcloud data.
"""

import json
import logging
import sys
from typing import Dict, List, Optional
from pathlib import Path

from policies.scene_graph import SceneGraph, Node, Edge, NodeLevel
from policies.utils.pointcloud_processor import PointcloudProcessor

# Modul-level Logger – gibt auf stderr aus, das ROS2 Launch weiterleitet
logger = logging.getLogger(__name__)


def _configure_logging():
    """
    Ensure log output is visible in ROS2 Launch terminal.
    """
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(name)s] [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


_configure_logging()


class FunGraphDatasetLoader:
    """Loads and parses FunGraph3D dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

        self.all_labels = self._load_json("all_labels.json")
        self.all_edges = self._load_json("all_edges.json")

        self.relations_by_scene = self._load_and_index_relations()
        self.annotations_by_scene = self._load_and_index_annotations()
        self.scene_list = self._load_scene_list()

        self._pointcloud_cache: Dict[
            str, Optional[PointcloudProcessor]
        ] = {}

    def _load_json(self, filename: str):
        filepath = self.dataset_path / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}"
            )
        with open(filepath, "r") as f:
            return json.load(f)

    def _load_and_index_relations(self) -> Dict[str, List[dict]]:
        relations = self._load_json("FunGraph3D.relations.json")
        indexed: Dict[str, List[dict]] = {}
        for relation in relations:
            scene_id = relation.get("scene_id")
            indexed.setdefault(scene_id, []).append(relation)
        return indexed

    def _load_and_index_annotations(self) -> Dict[str, List[dict]]:
        annotations = self._load_json(
            "FunGraph3D.annotations.json"
        )
        indexed: Dict[str, List[dict]] = {}
        for annotation in annotations:
            scene_id = annotation.get("scene_id")
            indexed.setdefault(scene_id, []).append(annotation)
        return indexed

    def _load_scene_list(self) -> List[str]:
        filepath = self.dataset_path / "OpenFunGraph_split.txt"
        if not filepath.exists():
            return []
        scene_ids = set()
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    scene_ids.add(line.split("/")[0])
        return sorted(list(scene_ids))

    def get_available_scenes(self) -> List[str]:
        return self.scene_list


    def _get_pointcloud(
        self, scene_id: str
    ) -> Optional[PointcloudProcessor]:
        """Load pointcloud for a scene with caching."""
        if scene_id in self._pointcloud_cache:
            cached = self._pointcloud_cache[scene_id]
            logger.debug(
                f"Cache hit for '{scene_id}': "
                f"{'found' if cached else 'not found'}"
            )
            return cached

        filepath = (self.dataset_path / scene_id).with_suffix(".ply")
        if filepath.exists():
            try:
                logger.info(f"  Loading: {filepath}")
                pc = PointcloudProcessor(str(filepath))
                logger.info(
                    f"  OK: {pc.num_points} points loaded"
                )
                self._pointcloud_cache[scene_id] = pc
                return pc
            except Exception as e:
                logger.error(
                    f"  FAILED to load {filepath}: {e}"
                )

        logger.warning(
            f"No PLY file found for scene '{scene_id}'. Path: '{filepath}'"
        )
        self._pointcloud_cache[scene_id] = None
        return None

    def load_scene(self, scene_id: str) -> SceneGraph:
        """Load a scene with pointcloud-based positions."""
        if scene_id not in self.annotations_by_scene:
            available = self.get_available_scenes()
            raise ValueError(
                f"Scene '{scene_id}' not found. "
                f"Available: {available}"
            )

        annotations = self.annotations_by_scene.get(
            scene_id, []
        )
        relations = self.relations_by_scene.get(scene_id, [])

        logger.info(
            f"Loading scene '{scene_id}': "
            f"{len(annotations)} annotations, "
            f"{len(relations)} relations"
        )

        pointcloud = self._get_pointcloud(scene_id)

        if pointcloud is not None:
            logger.info(
                f"Using pointcloud: {pointcloud.num_points} points"
            )
        else:
            logger.warning(
                f"No pointcloud for '{scene_id}'. "
                f"Falling back to lookat positions."
            )

        nodes = self._create_nodes(annotations, pointcloud)
        edges = self._create_edges(relations, annotations)

        return SceneGraph(
            scene_id=scene_id,
            nodes=nodes,
            edges=edges,
        )

    def _create_nodes(
        self,
        annotations: List[dict],
        pointcloud: Optional[PointcloudProcessor],
    ) -> List[Node]:
        """Create Node objects with positions from pointcloud."""
        nodes = []

        for idx, annotation in enumerate(annotations):
            node_id = annotation.get("annot_id", f"node_{idx}")
            label = annotation.get("label", "")
            indices = annotation.get("indices", [])

            position = None
            obb_data = None
            position_source = "none"

            # Priority 0: precomputed OBB from the annotation itself.
            # Written by scripts/fuse_scenes.py for ROOM nodes (room AABB),
            # which have no indices to derive geometry from.
            precomputed_obb = annotation.get("obb")
            if precomputed_obb is not None:
                obb_data = precomputed_obb
                position = list(precomputed_obb["center"])
                position_source = "precomputed_obb"

            # Priority 1: OBB center (falls back to centroid if OBB fails)
            if position is None and pointcloud is not None and indices:
                centroid = pointcloud.compute_centroid(indices)
                if centroid is not None:
                    obb_data = pointcloud.compute_obb(indices)
                    if obb_data is not None:
                        position = obb_data["center"]
                        position_source = "obb_center"
                    else:
                        position = centroid
                        position_source = "pointcloud_centroid"
                else:
                    logger.debug(
                        f"No centroid for node '{label}' "
                        f"({len(indices)} indices)"
                    )

            # Priority 2: lookat fallback
            if position is None:
                record_camera = annotation.get(
                    "record_camera", {}
                )
                lookat = record_camera.get("lookat")
                if lookat and len(lookat) == 3:
                    position = list(lookat)
                    position_source = "lookat_fallback"

            # Optional hierarchy level (BUILDING/ROOM/OBJECT). Flat FunGraph3D
            # scenes have no "level" key and stay None; fused multi-room scenes
            # set it so policies and the visualizer can tell rooms from objects.
            level_str = annotation.get("level")
            level = NodeLevel(level_str) if level_str else None

            node = Node(
                id=node_id,
                label=label,
                position=position,
                obb=obb_data,
                indices=indices,
                level=level,
            )
            nodes.append(node)

        return nodes

    def _create_edges(
        self,
        relations: List[dict],
        annotations: List[dict],
    ) -> List[Edge]:
        """Create Edge objects from relations."""
        edges = []

        ann_id_map = {
            ann.get("annot_id", ""): ann
            for ann in annotations
            if ann.get("annot_id")
        }

        for relation in relations:
            source_id = relation.get(
                "first_node_annot_id", ""
            )
            target_id = relation.get(
                "second_node_annot_id", ""
            )
            description = relation.get("description", "")

            if source_id in ann_id_map and target_id in ann_id_map:
                edge = Edge(
                    source_id=source_id,
                    target_id=target_id,
                    label=description,
                )
                edges.append(edge)

        return edges

    def get_scene_info(self, scene_id: str) -> Dict:
        """Get metadata about a scene."""
        if scene_id not in self.annotations_by_scene:
            raise ValueError(f"Scene '{scene_id}' not found")

        annotations = self.annotations_by_scene[scene_id]
        relations = self.relations_by_scene.get(scene_id, [])

        pointcloud = self._pointcloud_cache.get(scene_id)

        return {
            "scene_id": scene_id,
            "num_objects": len(annotations),
            "num_relations": len(relations),
            "object_labels": [
                ann.get("label", "") for ann in annotations
            ],
            "pointcloud_available": pointcloud is not None,
            "pointcloud_num_points": (
                pointcloud.num_points if pointcloud else 0
            ),
        }