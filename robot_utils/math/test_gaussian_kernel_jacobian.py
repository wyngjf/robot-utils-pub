import torch
from functools import partial
import numpy as np
from robot_utils.torch.autodiff import get_jacobian


def multivariate_gaussian_kernel(x, mu, sigma):
    """
    Computes the multivariate Gaussian kernel density for a given point `x`,
    with mean `mu` and covariance matrix `sigma`.
    """
    d = x.shape[-1]   # number of dimensions
    norm = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return norm.log_prob(x)


# Define the mean and covariance matrix of the Gaussian kernel function
mu = torch.tensor([1.0, 2.0], dtype=torch.float32)
sigma = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)

# Define the input point for which to compute the Jacobian
x = torch.tensor([[3.0, 4.0], [5., 6.]], dtype=torch.float32)
x.requires_grad = True   # enable automatic differentiation

# Compute the partial derivatives of the kernel density function with respect to each input variable
y = multivariate_gaussian_kernel(x, mu, sigma)
ic(x.shape, y.shape)
# jacobian = torch.autograd.grad(y, x, torch.eye(x.shape[-1]), create_graph=True)[0]
jacobian = get_jacobian(partial(multivariate_gaussian_kernel, mu=mu, sigma=sigma), x, output_dims=2)
ic(jacobian)
