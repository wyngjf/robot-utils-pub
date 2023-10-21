import numpy as np
from typing import Callable, Union

import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
# from mujoco_py.generated import const
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform
from robot_utils.math.transformations import euler_matrix, quaternion_matrix
from scipy.cluster.hierarchy import dendrogram
from robot_utils.py.const import *


# class Arrow(object):
#     __slots__ = ["pos", "size", "mat", "rgba", "geom_type", "label"]
#
#     def __init__(self):
#         self.pos = np.zeros(3)
#         self.size = np.zeros(3)
#         self.mat = np.zeros((3, 3))
#         self.rgba = np.ones(4)
#         self.geom_type = const.GEOM_ARROW,
#         self.label = "arrow"

def plt_full_screen():
    """ need to call this for every individual figure """
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

    def draw(self, renderer):
        FancyArrowPatch.draw(self, renderer)


def plot_mesh(x, y, ax=None, up_down_dir=True, cmap='Spectral', linewidths=0.5, **kwargs):
    """
    x, y are the x-/y-coordinates of the 2d grid. Both have dimensionality M x N
    Args:
        x (M x N): x coordinates
        y (M x N): y coordinates
        up_down_dir: set the color spectrum to change in the up-down direction. Set false will use left-right dir.
        ax (): axis to plot the data
        **kwargs ():

    Returns:

    """
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)            # (M, N, 2), each entry in the MxN matrix is a 2d coordinate (x, y)
    segs2 = segs1.transpose(1, 0, 2)            # (N, M, 2)

    # extract N-1 segments from each row and in total (M, N-1, 2, 2). each entry in the (M x N-1) matric is a line
    # segments containing two points [(x1, y1), (x2, y2)] (horizontal lines)
    segs1 = np.expand_dims(segs1, axis=-2)                          # (M, N, 1, 2)
    segs1 = np.concatenate([segs1[:, :-1], segs1[:, 1:]], axis=-2)  # (M, N-1, 2, 2)
    # this is for the vertical lines, similar as above
    segs2 = np.expand_dims(segs2, axis=-2)
    segs2 = np.concatenate([segs2[:, :-1], segs2[:, 1:]], axis=-2)

    # now, we concern the order of all the line segments to assign a correct color maps.
    if up_down_dir:
        # we consider the 2 horizontally neighboring segments are close and should use similar colors. So we
        # concatenate the vertical lines right after each row of horizontal lines.
        segs = np.concatenate([segs1[:-1], segs2.transpose(1, 0, 2, 3)], axis=1)
        segs = segs.reshape(-1, 2, 2)
        # append the last row of segs1 (so that the dimensionality of the matrices match)
        segs = np.concatenate([segs, segs1[-1]])
    else:
        segs = np.concatenate([segs1.transpose(1, 0, 2, 3), segs2[:-1]], axis=1)
        segs = segs.reshape(-1, 2, 2)
        segs = np.concatenate([segs, segs2[-1]])

    cm = plt.get_cmap(cmap)
    c = cm(np.linspace(0, 1, segs.shape[0]))

    ax.add_collection(LineCollection(segs, colors=c, linewidths=linewidths, **kwargs))
    ax.autoscale()


def draw_frame_2d(ax, config: np.ndarray, scale: float = 1.0, alpha: float = 0.5, arrow_list=None):
    """
    Args:
        ax: matplotlib axes to plot such frame
        config: configuration of the local frame, it can be multiple format, e.g. (1) [x, y, alpha], (2) 2x2 rotation matrix
        scale: the scaling of the arrow, default is 1.0
        alpha:
        arrow_list: a list of FancyArrow object, only used when you want to update the arrow data instead of creating new ones
    """
    arrow = np.array([[1, 0], [0, 1]]) * scale
    arrow_config = dict(
        alpha=alpha,
        width=0.03 * scale,
        head_width=0.03 * 4 * scale,
        length_includes_head=True,
        shape="full"
    )
    if len(config.shape) == 1:  # case: [x, y, alpha]
        alpha = config[2]
        rot_mat = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        arrow = np.einsum("ij, kj->ki", rot_mat, arrow)

    if arrow_list is None:
        arrow_list = []
        arrow_list.append(ax.arrow(config[0], config[1], arrow[0, 0], arrow[0, 1], color="blue", **arrow_config))
        arrow_list.append(ax.arrow(config[0], config[1], arrow[1, 0], arrow[1, 1], color="green", **arrow_config))
    else:
        arrow_list[0].set_data(x=config[0], y=config[1], dx=arrow[0, 0], dy=arrow[0, 1])
        arrow_list[1].set_data(x=config[0], y=config[1], dx=arrow[1, 0], dy=arrow[1, 1])
    return arrow_list


def draw_3d_frame_on_image(ax, rot_mat: np.ndarray, origin: np.ndarray, xyz_2_uv: Callable,
                           scaling: np.ndarray = None, alpha: float = 0.5, linewidths: float = 5, collections=None):
    """
    Args:
        ax: matplotlib axes to plot such frame
        rot_mat: 3x3 rotation matrix
        origin: the origin of the frame
        xyz_2_uv: a callable function that can convert your 3d coordinates to uv coordinates, which depends on your cameara params
        scaling: (3x1) array of the scaling of the arrows, default is 1.0
        alpha: alpha of the lines
        collections: a list of lineCollection object, only used when you want to update the arrow data instead of creating new ones
    """
    if scaling is None:
        scaling = np.ones(3, dtype=float).reshape(-1, 1)
    coordinate_pts_uv = xyz_2_uv(torch.from_numpy(origin + rot_mat.T * scaling).float()).numpy()
    arrow_line_collect = np.concatenate(
        (np.tile(xyz_2_uv(origin.reshape(1, -1)), (3, 1)), coordinate_pts_uv),
        axis=-1
    ).reshape((3, 2, 2)).tolist()
    if collections is None:
        ax.add_collection(LineCollection(arrow_line_collect, color=[cbf_r, cbf_g, cbf_b], linewidths=linewidths, alpha=alpha))
    else:
        collections.set_segments(arrow_line_collect)
    return collections


def draw_frame_3d(
        ax,
        config: np.ndarray,
        scale: float = 1.0,
        alpha: float = 1.0,
        collections=None,
        label: str = None,
        label_fontsize: int = 5,
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
    """
    arrow = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * scale
    arrow_start = np.zeros(3, dtype=float)

    length = np.size(config)

    if length == 6:  # case: [x, y, z, alpha, beta, gamma]
        arrow_start = config[:3]
        rot_mat = euler_matrix(*config[3:].tolist(), axes="sxyz")[:3, :3]
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
        rot_mat = quaternion_matrix(config, rot_only=True)
    elif length == 7:
        arrow_start = config[:3]
        rot_mat = quaternion_matrix(config[3:], rot_only=True)
    else:
        raise NotImplementedError
    arrow = np.einsum("ij, kj->ki", rot_mat, arrow)
    arrow_end = arrow + arrow_start
    arrow_line_collect = np.concatenate((np.tile(arrow_start, (3, 1)), arrow_end), axis=-1).reshape((3, 2, 3)).tolist()

    if collections is None:
        collections = ax.add_collection3d(
            Line3DCollection(arrow_line_collect, color=[cbf_r, cbf_g, cbf_b], linewidths=5, alpha=alpha, **kwargs)
        )
        if label is not None:
            ax.text(arrow_start[0], arrow_start[1], arrow_start[2]-0.1, fr'$F_{label}$', fontsize=label_fontsize)
            ax.text(arrow_end[0, 0], arrow_end[0, 1], arrow_end[0, 2]-0.1, fr'$x_{label}$', fontsize=label_fontsize)
            ax.text(arrow_end[1, 0], arrow_end[1, 1], arrow_end[1, 2]-0.1, fr'$y_{label}$', fontsize=label_fontsize)
            ax.text(arrow_end[2, 0], arrow_end[2, 1], arrow_end[2, 2]-0.1, fr'$z_{label}$', fontsize=label_fontsize)
    else:
        collections.set_segments(arrow_line_collect)

    return collections


def draw_vec_3d(ax, vec: np.ndarray, origin: Union[np.ndarray, None]=None, color=None, alpha: float = 1, scale: Union[float, np.ndarray] = 1., **kwargs):
    """
    Args:
        ax: plt axes
        vec: (3, ) or (n, 3) where n is the number of points
        origin: if specified (n, 3), it is used as the origin to plot the vector, otherwise use zero vec
        color:
        alpha:
        scale: float or (n, ) scaling of the vector
    """
    vec = vec.copy()
    if len(vec.shape) == 1:
        vec = vec.reshape(1, -1)
    if np.ndim(scale) == 1:
        scale = scale[np.newaxis, ...]
    vec *= scale
    n = vec.shape[0]
    if origin is None:
        origin = np.zeros((n, 3))
    if len(origin.shape) == 1:
        origin = origin.reshape(1, -1)
    arrow_line_collect = np.concatenate((origin, vec+origin), axis=-1).reshape((n, 2, 3))
    if color is None:
        ax.add_collection3d(Line3DCollection(arrow_line_collect, alpha=alpha, **kwargs))
    else:
        ax.add_collection3d(Line3DCollection(arrow_line_collect, color=color, alpha=alpha, **kwargs))


def plot_dendrogram(model, **kwargs):
    children = model.children_
    # The number of observations contained in each cluster level
    n_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, model.distances_, n_observations]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


def set_2d_equal_auto(ax, xlim=None, ylim=None):
    if xlim is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    d_x = xlim[1] - xlim[0]
    d_y = ylim[1] - ylim[0]
    max_d = 0.5 * max([d_x, d_y])
    mean_x = 0.5 * (xlim[0] + xlim[1])
    mean_y = 0.5 * (ylim[0] + ylim[1])
    ax.set_xlim(mean_x - max_d, mean_x + max_d)
    ax.set_ylim(mean_y - max_d, mean_y + max_d)


def set_3d_equal_auto(ax, xlim=None, ylim=None, zlim=None):
    ax.set_box_aspect((1, 1, 1))
    if xlim is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
    d_x = xlim[1] - xlim[0]
    d_y = ylim[1] - ylim[0]
    d_z = zlim[1] - zlim[0]
    max_d = 0.5 * max([d_x, d_y, d_z])
    mean_x = 0.5 * (xlim[0] + xlim[1])
    mean_y = 0.5 * (ylim[0] + ylim[1])
    mean_z = 0.5 * (zlim[0] + zlim[1])
    ax.set_xlim(mean_x - max_d, mean_x + max_d)
    ax.set_ylim(mean_y - max_d, mean_y + max_d)
    ax.set_zlim(mean_z - max_d, mean_z + max_d)


def set_3d_ax_label(ax, labels: list, **kwargs):
    ax.set_xlabel(labels[0], color=cbf_r, **kwargs)
    ax.set_ylabel(labels[1], color=cbf_g, **kwargs)
    ax.set_zlabel(labels[2], color=cbf_b, **kwargs)


def plot_2d_line_color_map(ax, line: np.ndarray, color_map: str = None, **kwargs):
    """

    Args:
        ax: the axes to plot the line
        line: the line to plot with shape (T, dim)
        color_map: the name of the color map
    """
    line = line[:, np.newaxis, :]
    segments = np.concatenate([line[:-1], line[1:]], axis=1)  # (T-1, 2, 2)
    if color_map:
        lc = LineCollection(segments, cmap=plt.get_cmap(color_map), **kwargs)
        lc.set_array(np.linspace(0, 1, line.shape[0]))
    else:
        lc = LineCollection(segments, **kwargs)
    ax.add_collection(lc)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    # grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 40))
    # plot_mesh(grid_x, grid_y, ax=ax, up_down_dir=False)
    colors = plt.get_cmap("plasma")(np.linspace(0, 1, 100))
    t = np.linspace(0, np.pi * 3, 100)
    traj = np.array((t, np.sin(t))).T
    plot_2d_line_color_map(ax, traj, color=colors)
    plt.autoscale()
    plt.show()
