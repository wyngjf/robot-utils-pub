"""
Note: to avoid circular dependency, this module should not depend on math/transformation
"""

import numpy as np


def sphere_logarithmic_map(x: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.
    Parameters
    ----------
    :param x: point on the manifold
    :param x0: basis point of the tangent space where x will be mapped

    Returns
    -------
    :return: u: vector in the tangent space of x0
    """
    # if np.ndim(x0) < 2:
    #     x0 = x0[:, None]
    #
    # if np.ndim(x) < 2:
    #     x = x[:, None]
    #
    # distance = np.arccos(np.clip(np.dot(x0.T, x), -1., 1.))
    #
    # u = (x - x0 * np.cos(distance)) * distance/np.sin(distance)
    # u[:, distance[0] < 1e-16] = np.zeros((u.shape[0], 1))

    if x0.ndim < 2:
        x0 = x0[np.newaxis, :]

    if x.ndim < 2:
        x = x[np.newaxis, :]

    distance = np.arccos(np.clip(np.einsum("bi,bi->b", x0, x), -1., 1.))[..., np.newaxis]
    u = (x - x0 * np.cos(distance)) * distance/np.sin(distance)
    u[distance.squeeze() < 1e-16, :] = np.zeros((1, u.shape[1]))
    return u.squeeze()