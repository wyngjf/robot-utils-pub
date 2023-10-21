import torch
from typing import Union


def random_uv_on_mask(
        img_mask_torch: torch.Tensor,
        num_samples: int,
        allow_replacement: bool = True
) -> Union[torch.Tensor, None]:
    """
    Args:
        img_mask_torch: (h, w)
        num_samples:
        allow_replacement: if there's not enough point within the mask compared to requested number, the allow replacement

    Returns: torch.LongTensor in (u, v) format. (n_samples, 2), same device as img_mask
    """
    mask_uv = torch.nonzero(img_mask_torch.transpose(1, 0))
    n_pixels = mask_uv.shape[0]
    if n_pixels == 0:
        return None

    num_samples_min = min(num_samples, n_pixels)
    indices = torch.randperm(n_pixels)[:num_samples_min].to(img_mask_torch.device)
    if num_samples > n_pixels and allow_replacement:
        complementary_idx = torch.randint(0, n_pixels, (num_samples - n_pixels,)).to(img_mask_torch.device)
        indices = torch.cat((indices, complementary_idx))
    return torch.index_select(mask_uv, 0, indices)


def random_uv(
        width: int, height: int, num_samples: int = 1, device: str = 'cpu'
) -> torch.Tensor:
    """
    return pixel uv coordinates (num_samples, 2)
    """
    two_rand_numbers = torch.rand(num_samples, 2)
    two_rand_numbers[:, 0] = two_rand_numbers[0, :] * width
    two_rand_numbers[:, 1] = two_rand_numbers[1, :] * height
    return torch.floor(two_rand_numbers).long().to(device)
