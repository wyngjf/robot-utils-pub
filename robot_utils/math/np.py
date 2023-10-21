import numpy as np


def get_mesh_grid(side_length, dim=2, range=None, reshape=True):
    """
    Generates a flattened grid of (x,y,...) coordinates
    Args:
        side_length: int or list/tuple of ints. int, generate same number of samples for each dim.
                     list, generate different number of samples for each dim
        dim: when side_length is int, you need to specify dimension of the coordinates
        range: a list of tuple, [(min, max), (min, max) ... ] specify the sample range of each dim

    Returns: flattened grid as 2D matrix, each row is a sampled coordinates

    """
    if isinstance(side_length, int):
        if range is None:
            tensors = tuple(dim * [np.linspace(-1, 1, side_length)])
        else:
            tensors = tuple(dim * [np.linspace(range[0], range[1], side_length)])
    else:
        if range is None:
            tensors = tuple([np.linspace(-1, 1, s) for s in side_length])
        else:
            tensors = tuple([np.linspace(r[0], r[1], s) for s, r in zip(side_length, range)])
    mesh_grid = np.stack(np.meshgrid(*tensors, indexing="ij"), axis=-1)
    if reshape:
        mesh_grid = mesh_grid.reshape(-1, dim)
    return mesh_grid
