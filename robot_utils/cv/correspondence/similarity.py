import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
    """
    compute the similarity of batches of vectors between a and b, where a and b may have different batches. If you
    want to compute cosine similarity of two matrices of the same size in a one-to-one corresponding way, use torch
    implementation: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

    Args:
        a: (batch_a, dim)
        b: (batch_b, dim)
        eps: for numerical stability

    Returns: the similarity matrix (batch_a, batch_b)

    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_matrix = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_matrix
