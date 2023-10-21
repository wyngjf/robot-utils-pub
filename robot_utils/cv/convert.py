import numpy as np
import torch


def uv_to_flattened_pixel_locations(u, v, image_width):
    """
    Converts to a flat tensor
    """
    return v * image_width + u


def flattened_pixel_locations_to_u_v(flat_pixel_locations, image_width):
    """
    :param flat_pixel_locations: A torch.LongTensor of shape torch.Shape([n,1]) where each element
     is a flattened pixel index, i.e. some integer between 0 and 307,200 for a 640x480 image

    :type flat_pixel_locations: torch.LongTensor

    :return: A tuple torch.LongTensor in (u,v) format
    the pixel and the second column is the v coordinate

    """
    return flat_pixel_locations % image_width, torch.div(flat_pixel_locations, image_width, rounding_mode="trunc")


def pil_image_to_cv2(pil_image):
    return np.array(pil_image)[:, :, ::-1].copy()
