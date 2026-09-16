from typing import List

from policies.dataset_adapters.base_adapter import DatasetAdapter
from policies.dataset_adapters.fungraph_loader import FunGraphDatasetLoader
from policies.scene_graph import SceneGraph


class FunGraph3DAdapter(DatasetAdapter):
    """
    Wrapper around FunGraphDatasetLoader.

    FunGraph3D is a flat dataset (no building/room/object hierarchy), so all
    nodes get level=None. It provides point cloud indices and bounding boxes
    for every annotated object, so the full visualisation pipeline is supported.
    """

    BUILD_GRAPH_AND_VISUALIZE = True

    def __init__(self, dataset_path: str):
        self._loader = FunGraphDatasetLoader(dataset_path)

    def get_available_scenes(self) -> List[str]:
        return self._loader.get_available_scenes()

    def load_scene(self, scene_id: str) -> SceneGraph:
        return self._loader.load_scene(scene_id)
