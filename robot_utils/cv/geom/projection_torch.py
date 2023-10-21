import torch
from typing import Union, Tuple


def pinhole_projection_image_to_camera(
        wh: torch.Tensor,
        z: torch.Tensor,
        intrinsic: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        wh: (2, ) or (n_points, 2) array of (width, height) coordinates of pixels
        z:  (h, w) or (h, w, n_depths) depth map
        intrinsic: (3, 3) intrinsic parameters

    Returns: the [x, y, z] coordinates of size (n_points, n_depth, 3) in world frame
    """
    if wh.ndim == 1:
        wh = wh.unsqueeze(0)
    if z.ndim == 2:
        z = z.unsqueeze(-1)

    whz = torch.einsum("bj,bk->bkj", torch.cat((wh, torch.ones((wh.shape[0], 1))), dim=-1), z[wh[:, 1], wh[:, 0]])
    xyz_camera = torch.einsum("ij,bkj->bki", torch.linalg.inv(intrinsic), whz)  # (b, n_depths, 3)
    return xyz_camera.squeeze()  # (b, n_depths, 3)


def pinhole_projection_camera_to_image(
        xyz_cam: torch.Tensor,
        intrinsic: torch.Tensor,
        return_z: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        xyz_world: (3, ) or (n_points, 3) or (n_points, n_depth, 3) the world coordinates
        intrinsic: camera intrinsic
        return_z: whether to return the computed z values in the new image

    Returns: the (width, height) coordinates of size (n_points, n_depth, 2).squeeze()

    """
    if xyz_cam.ndim == 1:
        xyz_cam = xyz_cam.unsqueeze(0).unsqueeze(0)
    elif xyz_cam.ndim == 2:
        xyz_cam = xyz_cam.unsqueeze(0)

    wh = torch.einsum("ij,bkj->bki", intrinsic, xyz_cam)  # (batch, n_depth, 3)
    z = wh[..., [2]]
    wh = (wh[..., :2] / z).long().squeeze(-2)
    if return_z:
        return wh, z.squeeze(-2).squeeze(-1)
    else:
        return wh


def pinhole_projection_image_to_world(
        wh: torch.Tensor,
        z: torch.Tensor,
        cam_pose: torch.Tensor,
        intrinsic: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        wh: (2, ) or (n_points, 2) array of (width, height) coordinates of pixels
        z:  (h, w) or (h, w, n_depths) depth map
        cam_pose: (4, 4) extrinsic (camera pose)
        intrinsic: (3, 3) intrinsic parameters

    Returns: the [x, y, z] coordinates of size (n_points, n_depth, 3) in world frame
    """
    if wh.ndim == 1:
        wh = wh.unsqueeze(0)
    if z.ndim == 2:
        z = z.unsqueeze(-1)

    whz = torch.einsum("bj,bk->bkj", torch.cat((wh, torch.ones((wh.shape[0], 1))), dim=-1), z[wh[:, 1], wh[:, 0]])
    # xyz_camera = np.einsum("ij,bkj->bki", inv(intrinsic), whz)                                # (b, n_depths, 3)
    # xyz_world = np.einsum("ji,bki->bkj", cam_pose[:3, :3], xyz_camera) + cam_pose[:3, 3]    # (b, n_depths, 3)

    xyz_world = torch.einsum("li,ij,bkj->bkl", cam_pose[:3, :3], torch.linalg.inv(intrinsic), whz) + cam_pose[:3, 3]
    return xyz_world  # (b, n_depths, 3)


def pinhole_projection_world_to_image(
        xyz_world: torch.Tensor,
        cam_pose: torch.Tensor,
        intrinsic: torch.Tensor,
        return_z: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        xyz_world: (3, ) or (n_points, 3) or (n_points, n_depth, 3) the world coordinates
        cam_pose: camera extrinsic
        intrinsic: camera intrinsic
        return_z: whether to return the computed z values in the new image

    Returns: the (width, height) coordinates of size (n_points, n_depth, 2).squeeze()

    """
    if xyz_world.ndim == 1:
        xyz_world = xyz_world.unsqueeze(0).unsqueeze(0)
    elif xyz_world.ndim == 2:
        xyz_world = xyz_world.unsqueeze(1)
    xyz_cam = torch.einsum("ij,bki->bkj", cam_pose[:3, :3], xyz_world - cam_pose[:3, 3])  # (b, n_depth, 3)

    wh = torch.einsum("ij,bkj->bki", intrinsic, xyz_cam)  # (batch, n_depth, 3)
    z = wh[..., [2]]
    wh = (wh[..., :2] / z).long().squeeze(-2)
    if return_z:
        return wh, z.squeeze(-2).squeeze(-1)
    else:
        return wh


def pinhole_projection_image_to_image(
        wh_a: torch.Tensor,
        z_A: torch.Tensor,
        cam_pose_a: torch.Tensor,
        cam_pose_b: torch.Tensor,
        intrinsic: torch.Tensor,
        return_z: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        wh_a: (n_points, 2)
        z_A:  (h, w, n_depths)
        cam_pose_a: (4, 4)
        cam_pose_b: (4, 4)
        intrinsic:  (3, 3)
        return_z: whether to return the computed z values in the new image

    Returns: (n_points, 2) each row as format (u, v) or (w, h)
    """
    return pinhole_projection_world_to_image(
        pinhole_projection_image_to_world(wh_a, z_A, cam_pose_a, intrinsic),
        cam_pose_b, intrinsic, return_z
    )


