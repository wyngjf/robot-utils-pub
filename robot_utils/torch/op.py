import torch
from typing import Tuple, Union, List
Shape = Union[List[int], Tuple[int, ...], torch.Size]


def ravel_index(coords: torch.Tensor, shape: Shape) -> torch.Tensor:
    """
    copied from https://github.com/francois-rozet/torchist/blob/master/torchist/__init__.py
    Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).
    """

    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def unravel_index(indices: torch.Tensor, shape: Shape) -> torch.Tensor:
    """
    copied from https://github.com/francois-rozet/torchist/blob/master/torchist/__init__.py
    Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]
