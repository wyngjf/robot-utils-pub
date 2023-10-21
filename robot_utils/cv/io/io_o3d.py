import numpy as np
import open3d as o3d
from pathlib import Path


def save_pcl_numpy_to_ply(pcl: np.ndarray, filename: Path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    acceptable_extension = [".ply", ".pcd"]
    if filename.suffix not in acceptable_extension:
        raise TypeError(f"the suffix of {filename} is not supported, please try one of {acceptable_extension}")
    o3d.io.write_point_cloud(str(filename), pcd, print_progress=True)
