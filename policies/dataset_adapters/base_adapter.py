"""Abstract base class for dataset adapters."""

from abc import ABC, abstractmethod
from typing import List

from policies.scene_graph import SceneGraph


class DatasetAdapter(ABC):
    """
    Convert a dataset's native format into the canonical SceneGraph.

    Each adapter is responsible for:
      - Parsing dataset files from dataset_path
      - Setting node.position (3-D centroid or equivalent), None if unavailable
      - Setting node.level (NodeLevel enum) for hierarchical datasets, None for flat ones
      - Setting node.obb and node.indices if the dataset provides them, else None

    BUILD_GRAPH_AND_VISUALIZE declares whether the dataset is complete enough
    to run the full pipeline (graph modification + RViz visualisation).
    """

    BUILD_GRAPH_AND_VISUALIZE: bool = False

    @abstractmethod
    def get_available_scenes(self) -> List[str]:
        """Return the list of scene identifiers available in the dataset."""
        ...

    @abstractmethod
    def load_scene(self, scene_id: str) -> SceneGraph:
        """
        Load and return the full SceneGraph for scene_id.

        The returned graph is the authoritative source for the graph modifier
        and visualiser. The graph builder will derive the minimal graph from it.
        """
        ...
