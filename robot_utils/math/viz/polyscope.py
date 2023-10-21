import numpy as np
import polyscope as ps
import open3d as o3d
from typing import Union
from robot_utils.py.utils import get_datetime_string
import robot_utils.math.transformations as tn
from robot_utils.py.const import *


def plot_quat(
        quat: np.ndarray,
        double_cover: bool = False,
        colors: np.ndarray = None,
        radius: float = None,
        name: str = None
):
    """

    Args:
        quat: (batch, 4)
        double_cover: True, copy data to antipodal points with the same color
        colors: (3, ), or (batch, 3) as RGB array
        radius: the radius of the point
        name: in order not to have name conflict, you need to have unique names. Default use current time.
    """
    if name is None:
        name = get_datetime_string()

    if np.ndim(colors) == 1:
        colors = np.tile(colors, (quat.shape[0], 1))
    if np.ndim(colors) == 2 and colors.shape[1] == 4:
        colors = colors[:, :3]

    if double_cover:
        quat = np.concatenate([quat, -quat], axis=0)
        colors = np.concatenate([colors, colors], axis=0)

    w_g_0 = np.where(quat[:, 0] > 0)[0]    # w > 0
    w_le_0 = np.where(quat[:, 0] <= 0)[0]  # w <= 0
    ic(w_g_0.shape, w_le_0.shape)

    d = np.array([2.5, 0, 0.])
    data1 = quat[w_g_0, 1:]
    data2 = quat[w_le_0, 1:] + d

    hemi1 = ps.register_point_cloud(
        name+"_hemi1",
        points=data1,
        enabled=True,
        radius=radius,
        transparency=0.5)

    hemi2 = ps.register_point_cloud(
        name+"_hemi2",
        points=data2,
        enabled=True,
        radius=radius,
        transparency=0.5)

    hemi1.add_color_quantity(name+"_c1", colors[w_g_0], enabled=True)
    hemi2.add_color_quantity(name+"_c2", colors[w_le_0], enabled=True)
    return hemi1, hemi2, w_g_0, w_le_0


def plot_vec_field(
        quat: np.ndarray,
        vec: np.ndarray,
        double_cover: bool = False,
        colors: np.ndarray = None,
        radius: float = None,
        name: str = None,
        length: float = 0.01,
        vector_type: str = "standard"
):
    """

    Args:
        quat: (batch, 4)
        vec: (batch, 4)
        double_cover: True, copy data to antipodal points with the same color
        colors: (3, ), or (batch, 3) as RGB array
        radius: the radius of the point
        name: in order not to have name conflict, you need to have unique names. Default use current time.
        length: the length of the vectors
        vector_type: "standard" or "ambient"

    Returns:

    """
    hemi1, hemi2, idx_1, idx_2 = plot_quat(quat, double_cover, colors, radius, name)
    hemi1.add_vector_quantity(f"{name}_vec2", vec[idx_1, 1:], enabled=True, vectortype=vector_type, length=length)
    hemi2.add_vector_quantity(f"{name}_vec2", vec[idx_2, 1:], enabled=True, vectortype=vector_type, length=length)
    return hemi1, hemi2, idx_1, idx_2


def plot_two_sphere(mesh_color: np.ndarray = None, edge_color: np.ndarray = None, name: str = None):
    if name is None:
        name = get_datetime_string()

    mesh = o3d.geometry.TriangleMesh().create_sphere(resolution=20)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)  # (V,3)
    faces = np.asarray(mesh.triangles)    # (F,3), the index of the points, by combining them we construct a face

    if mesh_color is None:
        mesh_color = (0.3, 0.6, 0.8)
    if edge_color is None:
        edge_color = ((0.3, 0.6, 0.8))
        # edge_color = ((0.8, 0.8, 0.8))

    s1 = ps.register_surface_mesh(
        f"{name}_ball1", vertices, faces,
        enabled=True,
        color=mesh_color,
        edge_color=edge_color,
        edge_width=1.0, smooth_shade=True,
        material='candy', transparency=0.2
    )

    vertices1 = vertices.copy()
    vertices1[:, 0] += 2.5
    s2 = ps.register_surface_mesh(
        f"{name}_ball2", vertices1, faces,
        enabled=True,
        color=mesh_color,
        edge_color=mesh_color,
        edge_width=1.0, smooth_shade=True,
        material='candy', transparency=0.2)
    return s1, s2


def draw_frame_3d(
        config: np.ndarray,
        scale: Union[float, np.ndarray] = 1.0,
        alpha: float = 1.0,
        collections=None,
        label: str = None,
        radius: float = 0.1,
        enabled: bool = True,
        **kwargs
):
    """
    Args:
        ax: matplotlib axes to plot such frame
        config: configuration of the local frame, it can be multiple format, e.g.
            - quaternion for orientation, the position [x, y, z] will be assumed to be zeros
            - position + quaternion [x, y, z, qw, qx, qy, qz], 7 DoF
            - position + flattened 3x3 rotation matrix, 3 + 9 = 12 DoF
            - position + RPY Euler angle, 6 DoF
        scale: the scaling of the arrow, default is 1.0
        alpha:
        collections:
        label: if None, no label will be added to the plot.
        radius: radius of the point and arrow
        enabled: set to True to visualize the frame, otherwise will be hidden
    """
    if label is None:
        label = get_datetime_string()
    if not isinstance(scale, np.ndarray):
        scale = np.ones(3, dtype=float) * scale
    arrow = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale
    arrow_start = np.zeros(3, dtype=float)

    length = np.size(config)

    if length == 6:  # case: [x, y, z, alpha, beta, gamma]
        arrow_start = config[:3]
        rot_mat = tn.euler_matrix(*config[3:].tolist(), axes="sxyz")[:3, :3]
    elif length == 12 and np.ndim(config) == 1:
        arrow_start = config[:3]
        rot_mat = config[3:].reshape(3, 3)
    elif length == 12 and np.ndim(config) == 2:
        arrow_start = config[:3, 3]
        rot_mat = config[:3, :3]
    elif length == 16:
        config = config.reshape(4, 4)
        arrow_start = config[:3, 3]
        rot_mat = config[:3, :3]
    elif length == 9:
        rot_mat = config.reshape(3, 3)
    elif length == 4:
        rot_mat = tn.quaternion_matrix(config, rot_only=True)
    elif length == 7:
        arrow_start = config[:3]
        rot_mat = tn.quaternion_matrix(config[3:], rot_only=True)
    else:
        raise NotImplementedError
    arrow = np.einsum("ij, kj->ki", rot_mat, arrow)

    if collections is None:
        collections = ps.register_point_cloud(
            f"{label}_ori", arrow_start[np.newaxis, ...], enabled=enabled, color=(1, 1, 1),
            transparency=alpha)
    else:
        collections.update_point_positions(arrow_start[np.newaxis, ...])
        collections.set_transparency(alpha)
        collections.remove_all_quantities()
        collections.set_enabled(enabled)

    collections.set_radius(radius * 0.2, relative=False)
    collections.add_vector_quantity(f"{label}_x", arrow[[0]], enabled=enabled, length=scale[0], color=cbf_r, radius=radius, vectortype="ambient")
    collections.add_vector_quantity(f"{label}_y", arrow[[1]], enabled=enabled, length=scale[1], color=cbf_g, radius=radius, vectortype="ambient")
    collections.add_vector_quantity(f"{label}_z", arrow[[2]], enabled=enabled, length=scale[2], color=cbf_b, radius=radius, vectortype="ambient")

    return collections
