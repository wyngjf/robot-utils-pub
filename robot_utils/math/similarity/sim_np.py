import numpy as np


def euc_l2_similarity_matching_row(a: np.array, b: np.array, wa: np.ndarray = None, wb: np.ndarray = None):
    dist = np.linalg.norm(a - b, axis=-1)
    if wa is not None and wb is not None:
        w = wa * wb
        dist = (dist * w).sum() / w.sum()
    return dist


def cosine_similarity_matching_rows(a: np.array, b: np.array, wa: np.ndarray = None, wb: np.ndarray = None, eps=1e-8):
    """
    compute the similarity of batches of vectors between a and b that has matching number rows (same batches)

    Args:
        a: (batch, dim)
        b: (batch, dim)
        wa: (batch_b, ) weights of a
        wb: (batch_b, ) weights of b
        eps: for numerical stability

    Returns: the similarity matrix (batch, )

    """
    a_n, b_n = np.linalg.norm(a, axis=1)[:, None], np.linalg.norm(b, axis=1)[:, None]
    a_norm = a / np.where(a_n < eps, eps, a_n)
    b_norm = b / np.where(b_n < eps, eps, b_n)
    sim_matrix = np.einsum("ni,ni->n", a_norm, b_norm)
    if wa is not None and wb is not None:
        w = wa * wb
        sim_matrix = (sim_matrix * w).sum() / w.sum()
    return sim_matrix


def cosine_similarity(a: np.array, b: np.array, eps=1e-8):
    """
    compute the similarity of batches of vectors between a and b, where a and b may have different batches.

    Args:
        a: (batch_a, dim)
        b: (batch_b, dim)
        eps: for numerical stability

    Returns: the similarity matrix (batch_a, batch_b)

    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / np.where(a_n < eps, eps, a_n)
    b_norm = b / np.where(b_n < eps, eps, b_n)
    return np.dot(a_norm, b_norm.T)
