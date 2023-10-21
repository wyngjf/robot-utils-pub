import numpy as np
from typing import Union, Tuple
from numpy.linalg import inv


def get_wh(width: int, height: int):
    ws = np.arange(width)
    hs = np.arange(height)
    hw = np.stack(np.meshgrid(hs, ws, indexing="ij"), axis=-1).reshape(-1, 2)
    return hw[:, [1, 0]].astype(int)


def pinhole_projection_image_to_camera(
        wh: np.ndarray,
        z: np.ndarray,
        intrinsic: np.ndarray
) -> np.ndarray:
    """
    Args:
        wh: (2, ) or (n_points, 2) array of (width, height) coordinates of pixels
        z:  (h, w) or (h, w, n_depths) depth map
        intrinsic: (3, 3) intrinsic parameters

    Returns: the [x, y, z] coordinates of size (n_points, n_depth, 3) in world frame
    """
    if wh.ndim == 1:
        wh = wh[np.newaxis, ...]
    if z.ndim == 2:
        z = z[..., np.newaxis]

    whz = np.einsum("bj,bk->bkj", np.concatenate((wh, np.ones((wh.shape[0], 1))), axis=-1), z[wh[:, 1], wh[:, 0]])
    xyz_camera = np.einsum("ij,bkj->bki", inv(intrinsic), whz)                                # (b, n_depths, 3)
    return xyz_camera.squeeze()  # (b, n_depths, 3)


def pinhole_projection_camera_to_image(
        xyz_cam: np.ndarray,
        intrinsic: np.ndarray,
        return_z: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Args:
        xyz_world: (3, ) or (n_points, 3) or (n_points, n_depth, 3) the world coordinates
        intrinsic: camera intrinsic
        return_z: whether to return the computed z values in the new image

    Returns: the (width, height) coordinates of size (n_points, n_depth, 2).squeeze()

    """
    if xyz_cam.ndim == 1:
        xyz_cam = xyz_cam[np.newaxis, ...][np.newaxis, ...]
    elif xyz_cam.ndim == 2:
        xyz_cam = xyz_cam[np.newaxis, ...]

    wh = np.einsum("ij,bkj->bki", intrinsic, xyz_cam)  # (batch, n_depth, 3)
    z = wh[..., [2]]
    wh = (wh[..., :2] / z).astype(int).squeeze(-2)
    if return_z:
        return wh, z.squeeze(-2).squeeze(-1)
    else:
        return wh


def pinhole_projection_image_to_world(
        wh: np.ndarray,
        z: np.ndarray,
        cam_pose: np.ndarray,
        intrinsic: np.ndarray
) -> np.ndarray:
    """
    Args:
        wh: (2, ) or (n_points, 2) array of (width, height) coordinates of pixels
        z:  (h, w) or (h, w, n_depths) depth map
        cam_pose: (4, 4) extrinsic (camera pose)
        intrinsic: (3, 3) intrinsic parameters

    Returns: the [x, y, z] coordinates of size (n_points, n_depth, 3) in world frame
    """
    if wh.ndim == 1:
        wh = wh[np.newaxis, ...]
    if z.ndim == 2:
        z = z[..., np.newaxis]

    whz = np.einsum("bj,bk->bkj", np.concatenate((wh, np.ones((wh.shape[0], 1))), axis=-1), z[wh[:, 1], wh[:, 0]])
    # xyz_camera = np.einsum("ij,bkj->bki", inv(intrinsic), whz)                                # (b, n_depths, 3)
    # xyz_world = np.einsum("ji,bki->bkj", cam_pose[:3, :3], xyz_camera) + cam_pose[:3, 3]    # (b, n_depths, 3)

    xyz_world = np.einsum("li,ij,bkj->bkl", cam_pose[:3, :3], inv(intrinsic), whz) + cam_pose[:3, 3]
    return xyz_world  # (b, n_depths, 3)


def pinhole_projection_world_to_image(
        xyz_world: np.ndarray,
        cam_pose: np.ndarray,
        intrinsic: np.ndarray,
        return_z: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Args:
        xyz_world: (3, ) or (n_points, 3) or (n_points, n_depth, 3) the world coordinates
        cam_pose: camera extrinsic
        intrinsic: camera intrinsic
        return_z: whether to return the computed z values in the new image

    Returns: the (width, height) coordinates of size (n_points, n_depth, 2).squeeze()

    """
    if xyz_world.ndim == 1:
        xyz_world = xyz_world[np.newaxis, np.newaxis, ...]
    elif xyz_world.ndim == 2:
        xyz_world = xyz_world[np.newaxis, ...]
    xyz_cam = np.einsum("ij,bki->bkj", cam_pose[:3, :3], xyz_world - cam_pose[:3, 3])  # (b, n_depth, 3)

    wh = np.einsum("ij,bkj->bki", intrinsic, xyz_cam)  # (batch, n_depth, 3)
    z = wh[..., [2]]
    wh = (wh[..., :2] / z).astype(int).squeeze(-2)
    if return_z:
        return wh, z.squeeze(-2).squeeze(-1)
    else:
        return wh


def pinhole_projection_image_to_image(
        wh_a: np.ndarray,
        z_A: np.ndarray,
        cam_pose_a: np.ndarray,
        cam_pose_b: np.ndarray,
        intrinsic: np.ndarray,
        return_z: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Args:
        wh_a: (n_points, 2)
        z_A:  (h, w, n_depths)
        cam_pose_a: (4, 4)
        cam_pose_b: (4, 4)
        intrinsic:  (3, 3)
        return_z: whether to return the computed z values in the new image

    Returns: (n_points, 2)
    """
    return pinhole_projection_world_to_image(
        pinhole_projection_image_to_world(wh_a, z_A, cam_pose_a, intrinsic),
        cam_pose_b, intrinsic, return_z
    )


