from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from robot_utils.cv.image.random_torch import random_uv_on_mask, random_uv
from robot_utils.cv.geom.projection_torch import pinhole_projection_world_to_image, pinhole_projection_image_to_world
from robot_utils.torch.op import unravel_index, ravel_index
from robot_utils.cv.correspondence.similarity import cosine_similarity


def pixel_correspondences(
        img_a_depth: np.ndarray, img_a_pose: np.ndarray,
        img_b_depth: np.ndarray, img_b_pose: np.ndarray,
        intrinsic: np.ndarray,
        uv_a: np.ndarray = None,
        n_samples: int = 20,
        device: str = 'cpu',
        img_a_mask: np.ndarray = None,
        img_b_mask: np.ndarray = None,
        ignore_fov_occlusion: bool = False,
        nearest_depth: float = 0.1,
        occlusion_margin: float = 0.003,
        lower: torch.Tensor = None,
        upper: torch.Tensor = None,
        rotation: torch.Tensor = None,
):
    """
    Computes pixel correspondences in batch
    Args:
        img_a_depth: (h, w)
        img_a_pose:  (4, 4)
        img_b_depth: (h, w)
        img_b_pose:  (4, 4)
        uv_a: torch.Long (2, ) or (n_samples, 2). pixel positions for which to find matches
        n_samples:
        device: 'cpu' or 'cuda'
        img_a_mask: (h, w) an image where each nonzero pixel will be used as a mask
        img_b_mask: (h, w) an image where each nonzero pixel will be used as b mask
        intrinsic: (3, 3), intrinsic matrix
        ignore_fov_occlusion:
        nearest_depth: the nearest depth allowed in a depth map
        occlusion_margin:
        lower: lower bound of the aabb in nerf
        upper: upper bound of the aabb in nerf
        rotation: rotation matrix of the aabb in nerf world coordinate system

    Returns: "Tuple None" if not found else "Tuple of torch.LongTensor", (uv_a, uv_b). Each has shape (n_sample, 2)
    """
    assert (img_a_depth.shape == img_b_depth.shape)
    image_height, image_width = img_a_depth.shape[:2]

    if img_a_mask is None:
        if uv_a is None:
            uv_a = random_uv(width=image_width, height=image_height, num_samples=n_samples)
        else:
            uv_a = torch.from_numpy(uv_a).long().to(device)
    else:
        img_a_mask = torch.from_numpy(img_a_mask).to(device)
        uv_a = random_uv_on_mask(img_a_mask, num_samples=n_samples)   # (num_attempts, 2)
        if uv_a is None:
            return None, None

    # Step: 1) find non-zero depths
    img_a_depth_torch = torch.from_numpy(img_a_depth).to(device).float()
    img_b_depth_torch = torch.from_numpy(img_b_depth).to(device).float()
    nonzero_indices = torch.where(img_a_depth_torch[uv_a[:, 1], uv_a[:, 0]] > nearest_depth)[0]
    uv_a = uv_a[nonzero_indices]

    # Step: 2) projection
    img_a_pose = torch.from_numpy(img_a_pose).to(device).float()
    img_b_pose = torch.from_numpy(img_b_pose).to(device).float()
    intrinsic = torch.from_numpy(intrinsic).to(device).float()
    xyz_b_world = pinhole_projection_image_to_world(uv_a, img_a_depth_torch, img_a_pose, intrinsic).squeeze(1)
    if lower is not None and upper is not None and rotation is not None:
        pcl = torch.einsum("ij,bj->bi", rotation, xyz_b_world)
        within_aabb_indices = torch.where(torch.all(torch.logical_and(lower <= pcl, upper >= pcl), dim=1))[0]
        if within_aabb_indices.nelement() == 0:
            return None, None
        uv_a, xyz_b_world = uv_a[within_aabb_indices], xyz_b_world[within_aabb_indices]

    uv_b, z_b = pinhole_projection_world_to_image(
        xyz_b_world, img_b_pose, intrinsic, return_z=True
    )
    uv_b = uv_b.squeeze(1)

    if ignore_fov_occlusion:
        return uv_a, uv_b

    # Step: 3) Field of View filter
    fov_indices = torch.where(
        (uv_b[..., 0] >= 0) & (uv_b[..., 0] < image_width) & (uv_b[..., 1] >= 0) & (uv_b[..., 1] < image_height)
    )[0]
    if fov_indices.nelement() == 0:
        return None, None
    uv_a, uv_b, z_b = uv_a[fov_indices], uv_b[fov_indices], z_b[fov_indices]

    # Step: 2.2) filter out points out side of bounding box
    if img_b_mask is not None:
        img_b_mask = torch.from_numpy(img_b_mask).to(device)
        uv_b_in_mask_indices = torch.where(img_b_mask[uv_b[..., 1], uv_b[..., 0]])[0]
        if uv_b_in_mask_indices.nelement() == 0:
            return None, None
        uv_a, uv_b, z_b = uv_a[uv_b_in_mask_indices], uv_b[uv_b_in_mask_indices], z_b[uv_b_in_mask_indices]

    # Step: 4) occlusion filter
    non_occlude_indices = torch.where((z_b - occlusion_margin) < img_b_depth_torch[uv_b[:, 1], uv_b[:, 0]])[0]
    if non_occlude_indices.nelement() == 0:
        return None, None
    else:
        return uv_a[non_occlude_indices], uv_b[non_occlude_indices]


def create_non_correspondences(
        uv_b_matches: torch.Tensor,
        img_b_shape,
        num_non_matches_per_match: int = 100,
        img_b_mask: Union[np.ndarray, None] = None,
        threshold: int = 100,
        within_mask: bool = True,
):
    """
    Takes in pixel matches (uv_b_matches) that correspond to matches in another image,
    and generates non-matches by just sampling in image space.
    Optionally, the non-matches can be sampled from a mask for image b.

    Args:
        uv_b_matches: (num_matches, 2) as (u, v) coordinates
        img_b_shape: (h, w)
        num_non_matches_per_match:
        img_b_mask: (h, w)
        threshold: if distance in pixel is smaller than threshold, we consider two pixel too close
        within_mask: True to sample non-match points within the object mask, otherwise, sample points off masks

    Returns: (num_matches, num_non_matches_per_match, 2) non-match uv coordinates, same device as uv_b_matches

    """
    if uv_b_matches is None:
        return None
    if isinstance(uv_b_matches, np.ndarray):
        uv_b_matches = torch.from_numpy(uv_b_matches)
    device = uv_b_matches.device
    h, w = img_b_shape[:2]
    num_matches = uv_b_matches.shape[0]
    n_sample = num_matches * num_non_matches_per_match

    # Step: 1) random sample non-match uv
    if img_b_mask is None:
        return random_uv(w, h, n_sample).reshape((num_matches, num_non_matches_per_match, 2)).to(device)
    else:
        if within_mask:
            mask = torch.from_numpy(img_b_mask).to(device)
        else:
            mask = torch.logical_not(torch.from_numpy(img_b_mask)).to(device)
        if torch.nonzero(mask).shape[0] == 0:
            return random_uv(w, h, n_sample).reshape((num_matches, num_non_matches_per_match, 2)).to(device)
        else:
            uv_non_match = random_uv_on_mask(mask, n_sample).reshape((num_matches, num_non_matches_per_match, 2))

    # Step: 2) compute distance to match uv and apply purturbation in pixel space if non-matches are too close to matches
    distance = torch.linalg.norm((uv_b_matches.unsqueeze(1) - uv_non_match).float(), dim=-1)
    perturb_idx = torch.where(distance < threshold)  # (idx_match, idx_non_match) to be perturbed
    if perturb_idx[0].shape[0] > 0:
        size = (perturb_idx[0].shape[0], 2)
        sign = (torch.rand(size, device=device) > 0.5) * 2 - 1
        perturb = torch.randint(threshold, threshold + 50, size, device=device) * sign
        uv_non_match[perturb_idx[0], perturb_idx[1]] += perturb
    return torch.clamp(uv_non_match, torch.zeros(2, device=device), torch.tensor([w, h], device=device)).long()


def find_best_match_for_descriptor(descriptor: torch.Tensor, descriptor_image: torch.Tensor, mask: torch.Tensor = None):
    """
    Compute the correspondences between the given descriptor and the descriptor image
    Args:
        descriptor: (D, ) or (*, D)
        descriptor_image: (h, w, D)
        mask: (h, w)

    Returns: (best_match_uv, best_match_diff, norm_diffs) as torch.Tensor
    best_match_idx is again in (u,v) = (right, down) coordinates

    """
    if descriptor.ndim == 1:
        descriptor = descriptor.unsqueeze(0)
    b = descriptor.shape[0]

    norm_diffs = torch.linalg.norm(descriptor_image.unsqueeze(2) - descriptor, axis=-1)  # (h, w, b)
    if mask is not None:
        norm_diffs_masked = torch.masked.masked_tensor(norm_diffs, mask.unsqueeze(-1).repeat(1, 1, b))
        best_match_flattened_idx = norm_diffs_masked.reshape(-1, b).argmin(dim=0)   # int as flattened index to a pixel
        best_match_flattened_idx = best_match_flattened_idx.get_data().long()
    else:
        best_match_flattened_idx = norm_diffs.reshape(-1, b).argmin(dim=0)  # int as flattened index to a pixel

    best_match_xy = unravel_index(best_match_flattened_idx, norm_diffs.shape[:2])  # (b, 2)
    best_match_xy_batch = torch.cat((best_match_xy, torch.arange(b).reshape(-1, 1).to(best_match_xy)), dim=-1)  # (b, 3)
    best_match_xy_batch = ravel_index(best_match_xy_batch, norm_diffs.shape)  # (b, )
    best_match_diff = norm_diffs.take(best_match_xy_batch)

    return best_match_xy[:, [1, 0]].squeeze(0), best_match_diff, norm_diffs.squeeze(-1)


def find_best_match_for_descriptor_cosine(
        descriptor: torch.Tensor,
        descriptor_image: torch.Tensor,
        mask: torch.Tensor = None,
        is_softmax: bool = False
):
    """
    Compute the correspondences between the given descriptor and the descriptor image
    Args:
        descriptor: (D, ) or (*, D)
        descriptor_image: (h, w, D)
        mask: (h, w)
        is_softmax: if True, use softmax to compute the probability of each row of the similarity matrix and
            take the expectation as the index of best match. Otherwise, use argmin.

    Returns: best match uv in torch.Long

    """
    if descriptor.ndim == 1:
        descriptor = descriptor.unsqueeze(0)

    mask_uv = torch.vstack(torch.where(mask)[::-1]).transpose(1, 0)
    descriptor_from_image = descriptor_image[mask_uv[:, 1], mask_uv[:, 0]]

    sim_matrix = cosine_similarity(descriptor, descriptor_from_image)  # (n, m)
    if is_softmax:
        prob = F.softmax(sim_matrix, dim=-1)
        return torch.round(torch.einsum("nm,mi->ni", prob, mask_uv.float())).long().squeeze(0)

    best_match_idx = torch.argmax(sim_matrix, dim=-1)
    best_match_uv = mask_uv[best_match_idx]
    return best_match_uv.squeeze(0)
