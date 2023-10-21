import numpy as np
import torch
from typing import Tuple, List, Union
from pathlib import Path
from rich.progress import track
from torchvision import transforms
from robot_utils.cv.io.io_torch import load_rgb


def flatten_uv_tensor(uv_tensor: torch.LongTensor, image_width: int):
    """
    Flattens a uv_tensor to single dimensional tensor
    Args:
        uv_tensor: (n_samples, 2) each row (u, v)
        image_width:

    Returns: (n_samples, ) each element v * w + u

    """
    return uv_tensor[:, 1] * image_width + uv_tensor[:, 0]


def unflatten_uv_tensor(uv_tensor: torch.LongTensor, image_width: int):
    """
     Flattens a 1-dim uv_tensor to 2-dim
    Args:
        uv_tensor: (n_samples, ) each element v * w + u
        image_width:

    Returns: (n_samples, 2) each row (u, v)

    """
    return torch.cat((
        (uv_tensor % image_width).long(),
        torch.div(uv_tensor, image_width, rounding_mode='floor').long()
    )).reshape(2, -1).transpose(1, 0)


def compute_image_mean_and_std_dev(image_filename_list: List[Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the image_mean and std_dev
    Args:
        image_filename_list: a list of image files to use for the calculation, absolute path to image file

    Returns: mean (3, ), standard deviation (3, )
    """
    num_image_samples = len(image_filename_list)
    if num_image_samples == 0:
        raise FileNotFoundError("no images in your list")
    ic(image_filename_list)
    mean = torch.zeros(3, dtype=torch.float32)
    std = torch.zeros(3, dtype=torch.float32)

    for f in track(image_filename_list, description="reading images"):
        if not f.is_file():
            raise FileNotFoundError(f"{f} does not exist")
        img = load_rgb(f, to_float=True)
        mean += torch.mean(img, dim=[-2, -1])
        std += torch.std(img, dim=[-2, -1])

    mean /= num_image_samples
    std /= num_image_samples
    return mean, std


def get_imagenet_normalization():
    """
    create a torchvision transform to normalize images with ImageNet mean and std
    """
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    img_std_dev = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    return transforms.Normalize(img_mean, img_std_dev)


class ResizedCrop:
    def __init__(
            self,
            orig_size_hw: Union[tuple, list] = None,
            new_size_hw: Union[tuple, list] = None,
            with_to_tensor: bool = True,
            with_imagenet_norm: bool = True,
            respect_hw_ratio: bool = True,
    ):
        transform_list = []
        if with_to_tensor:
            transform_list.append(transforms.ToTensor())
        if with_imagenet_norm:
            transform_list.append(get_imagenet_normalization())

        self.scale: Union[float, np.ndarray] = 1.0
        if respect_hw_ratio:
            if orig_size_hw is not None and new_size_hw is not None:
                orig_hw_ratio = orig_size_hw[0] / orig_size_hw[1]
                new_hw_ratio = new_size_hw[0] / new_size_hw[1]
                if orig_hw_ratio > new_hw_ratio:
                    crop_size = (int(orig_size_hw[1]) * new_hw_ratio, orig_size_hw[1])
                else:
                    crop_size = (orig_size_hw[0], int(orig_size_hw[0] / new_hw_ratio))
                transform_list.append(transforms.CenterCrop(size=crop_size))
                transform_list.append(transforms.Resize(size=new_size_hw))
                self.scale = new_size_hw[0] / crop_size[0]
        else:
            transform_list.append(transforms.Resize(size=new_size_hw))
            self.scale = (np.array(new_size_hw) / np.array(orig_size_hw))[::-1].reshape(2, 1)  # shape (2,1) with (w,h)
        self.transform = transforms.Compose(transform_list)

    def forward(self, x: np.ndarray):
        return self.transform(x)

    def __call__(self, x: np.ndarray, *args, **kwargs):
        return self.forward(x)

    def adapt_intrinsics(self, intrinsics: Union[np.ndarray, list]):
        if isinstance(intrinsics, list):
            intrinsics = np.array(intrinsics)
        intrinsics[:2] *= self.scale
        return intrinsics
