import torch
from torch import autograd


def get_jacobian(model, x, output_dims, reshape_flag=True):
	"""
	to compute the jacobian of model w.r.t. input variable x.
	x: (batch, dim), output_dims = 2, x_m: (2, 2), y_m: (2, 2)
	"""
	if x.ndimension() == 1:
		n = 1
	else:
		n = x.size()[0]
	x_m = x.repeat(1, output_dims).view(-1, output_dims)
	x_m.requires_grad_(True)
	y_m = model(x_m)
	if isinstance(y_m, tuple):
		y_m = y_m[0]
	mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
	# y.backward(mask)
	J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
	if reshape_flag:
		J = J.reshape(n, output_dims, output_dims)
	return J