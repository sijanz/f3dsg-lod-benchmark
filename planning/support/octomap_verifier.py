"""
OctoMap Leaf Verifier Utility.

Verifies resident OctoMap leaf counts from ROS 2 Octomap messages using
liboctomap_bridge.so.
"""

import ctypes
import os
from typing import Optional

_so_path = os.path.join(os.path.dirname(__file__), "liboctomap_bridge.so")
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(_so_path):
            raise FileNotFoundError(f"OctoMap bridge library not found at {_so_path}")
        _lib = ctypes.CDLL(_so_path)
        _lib.octree_read_binary_data.argtypes = [ctypes.c_double, ctypes.c_char_p, ctypes.c_size_t]
        _lib.octree_read_binary_data.restype = ctypes.c_void_p
        _lib.octree_get_num_leaf_nodes.argtypes = [ctypes.c_void_p]
        _lib.octree_get_num_leaf_nodes.restype = ctypes.c_size_t
        _lib.octree_free.argtypes = [ctypes.c_void_p]
        _lib.octree_free.restype = None
    return _lib


def get_resident_octree_leaf_count(octomap_msg) -> int:
    """
    Given an octomap_msgs/msg/Octomap message (e.g. from GetPlanningScene),
    extract the raw binary data bytes and parse via OcTree::readBinaryData().
    Returns leaf count as an int.
    """
    data_bytes = bytes(octomap_msg.data)
    if not data_bytes:
        return 0
    res = float(octomap_msg.resolution)
    lib = _get_lib()
    tree_ptr = lib.octree_read_binary_data(ctypes.c_double(res), data_bytes, len(data_bytes))
    if not tree_ptr:
        raise RuntimeError("Failed to deserialize octree from raw binary data bytes.")
    try:
        count = int(lib.octree_get_num_leaf_nodes(tree_ptr))
    finally:
        lib.octree_free(tree_ptr)
    return count
