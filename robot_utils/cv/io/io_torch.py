from typing import Union
from pathlib import Path

import torch
from torchvision.io import read_image


def load_rgb(filename: Union[str, Path], to_float: bool = False, **kwargs) -> torch.Tensor:
    """
    Args:
        filename:
        to_float: convert to float tensor
        **kwargs: e,g, mode=torchvision.io.ImageReadMode.UNCHANGED

    Returns: (c, h, w) as uint8 [0, 255]

    """
    if to_float:
        return torch.true_divide(read_image(str(filename), **kwargs), 255.)
    else:
        return read_image(str(filename), **kwargs)

