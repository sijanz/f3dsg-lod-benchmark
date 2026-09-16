"""
Pointcloud Processor Utility.

Utility for loading 3D point cloud meshes (.ply) and computing spatial properties
such as centroids and bounding boxes for node annotations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any


class PointcloudProcessor:
    """Processes 3D point cloud files for scene graph nodes."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.points: List[List[float]] = []
        self._load_points()

    @property
    def num_points(self) -> int:
        return len(self.points)

    def _load_points(self):
        if not self.filepath.exists():
            raise FileNotFoundError(f"Point cloud file not found: {self.filepath}")

        # Try open3d if available
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(self.filepath))
            pts = [list(pt) for pt in pcd.points]
            if len(pts) > 0:
                self.points = pts
                return
        except ImportError:
            pass

        # Try trimesh if available
        try:
            import trimesh
            mesh = trimesh.load(str(self.filepath))
            if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
                self.points = [list(v) for v in mesh.vertices]
                return
        except ImportError:
            pass

        # Try plyfile if available
        try:
            from plyfile import PlyData
            plydata = PlyData.read(str(self.filepath))
            vertex = plydata["vertex"]
            x = vertex["x"]
            y = vertex["y"]
            z = vertex["z"]
            self.points = [[float(x[i]), float(y[i]), float(z[i])] for i in range(len(x))]
            return
        except ImportError:
            pass

        # Fallback: simple parser for basic PLY files
        try:
            with open(self.filepath, "r", errors="ignore") as f:
                header = True
                pts = []
                for line in f:
                    if header:
                        if line.strip() == "end_header":
                            header = False
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            continue
                if pts:
                    self.points = pts
        except Exception:
            pass

    def compute_centroid(self, indices: List[int]) -> Optional[List[float]]:
        """Compute the 3D arithmetic mean of points at the specified indices."""
        if not indices or not self.points:
            return None
        valid_pts = [self.points[idx] for idx in indices if 0 <= idx < len(self.points)]
        if not valid_pts:
            return None
        n = len(valid_pts)
        cx = sum(p[0] for p in valid_pts) / n
        cy = sum(p[1] for p in valid_pts) / n
        cz = sum(p[2] for p in valid_pts) / n
        return [float(cx), float(cy), float(cz)]

    def compute_obb(self, indices: List[int]) -> Optional[Dict[str, Any]]:
        """Compute bounding box for the subset of points."""
        if not indices or not self.points:
            return None
        valid_pts = [self.points[idx] for idx in indices if 0 <= idx < len(self.points)]
        if not valid_pts:
            return None
        xs = [p[0] for p in valid_pts]
        ys = [p[1] for p in valid_pts]
        zs = [p[2] for p in valid_pts]
        min_pt = [min(xs), min(ys), min(zs)]
        max_pt = [max(xs), max(ys), max(zs)]
        center = [(min_pt[0] + max_pt[0]) / 2.0, (min_pt[1] + max_pt[1]) / 2.0, (min_pt[2] + max_pt[2]) / 2.0]
        extent = [max_pt[0] - min_pt[0], max_pt[1] - min_pt[1], max_pt[2] - min_pt[2]]
        return {
            "center": center,
            "extent": extent,
            "min": min_pt,
            "max": max_pt,
        }

