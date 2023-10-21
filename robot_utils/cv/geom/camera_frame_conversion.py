import numpy as np


# ============================== Nerf <--> Colmap/OpenCV ==============================
def pose_nerf_to_colmap(
        pose: np.ndarray,
        data_rotation: np.ndarray,
        data_offset: np.ndarray,
        data_scale: float
):
    mat = np.copy(pose)
    mat[0:3, 3] /= data_scale
    mat[0:3, 3] += data_offset

    R_inv = np.linalg.inv(data_rotation)
    mat = R_inv @ mat

    mat[2, :] *= -1
    mat = mat[[1, 0, 2, 3], :]
    mat[:3, 1] *= -1
    mat[:3, 2] *= -1
    return mat


def pose_colmap_to_nerf(
        pose: np.ndarray,
        data_rotation: np.ndarray,
        data_offset: np.ndarray,
        data_scale: np.ndarray
):
    mat = np.copy(pose)

    mat[:3, 1] *= -1
    mat[:3, 2] *= -1
    mat = mat[[1, 0, 2, 3], :]
    mat[2, :] *= -1

    mat = data_rotation @ mat

    mat[0:3, 3] -= data_offset
    mat[0:3, 3] *= data_scale
    return mat


# ============================== Nerf <--> NGP ==============================
def pose_ngp_to_nerf(
        pose: np.ndarray,
        ngp_offset: np.ndarray = np.array([0.5, 0.5, 0.5]),
        ngp_scale: float = 0.33
):
    mat = np.copy(pose)
    mat[[0, 1, 2], :] = mat[[2, 0, 1], :]  # swap axis

    mat[:, 1] *= -1  # flip axis
    mat[:, 2] *= -1

    if pose.size == 16:
        mat[:3, 3] -= ngp_offset  # translation and re-scale
        mat[:3, 3] /= ngp_scale

    return mat


def pose_nerf_to_ngp(
        pose: np.ndarray,
        ngp_offset: np.ndarray = np.array([0.5, 0.5, 0.5]),
        ngp_scale: float = 0.33,
):
    mat = np.copy(pose)
    mat = mat[:-1, :]

    mat[:, 1] *= -1  # flip axis
    mat[:, 2] *= -1

    if np.size(pose) == 16:
        # ic(scale, translation)
        mat[:3, 3] *= ngp_scale
        mat[:3, 3] += ngp_offset  # translation and re-scale

    mat[[0, 1, 2], :] = mat[[1, 2, 0], :]
    return mat


# ============================== OpenGL <--> Colmap/OpenCV ==============================
def pose_colmap_to_opengl(pose: np.ndarray):
    # cam_pose = pose.copy()
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    cam_pose = np.matmul(pose, flip_yz)
    return cam_pose


# ============================== OpenGL <--> NGP ==============================
def pose_ngp_to_opengl(
        pose: np.ndarray,
        data_rotation: np.ndarray,
        data_offset: np.ndarray,
        data_scale: float,
        ngp_offset: np.ndarray,
        ngp_scale: float
):
    pose_nerf = pose_ngp_to_nerf(pose, ngp_offset, ngp_scale)
    pose_colmap = pose_nerf_to_colmap(pose_nerf, data_rotation, data_offset, data_scale)
    return pose_colmap_to_opengl(pose_colmap)


# ============================== OpenGL <--> NGP ==============================
def pose_ngp_to_colmap(
        pose: np.ndarray,
        data_rotation: np.ndarray,
        data_offset: np.ndarray,
        data_scale: float,
        ngp_offset: np.ndarray,
        ngp_scale: float
):
    pose_nerf = pose_ngp_to_nerf(pose, ngp_offset, ngp_scale)
    pose_colmap = pose_nerf_to_colmap(pose_nerf, data_rotation, data_offset, data_scale)
    return pose_colmap


# ============================== OpenGL <--> Nerf ==============================
def pose_nerf_to_opengl(
        pose: np.ndarray,
        data_rotation: np.ndarray,
        data_offset: np.ndarray,
        data_scale: float
):
    pose_colmap = pose_nerf_to_colmap(pose, data_rotation, data_offset, data_scale)
    return pose_colmap_to_opengl(pose_colmap)


