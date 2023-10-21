import copy

import cv2
import numpy as np
from typing import Union, List


def draw_reticle(img_raw, u, v, label_color, alpha = 1.0, scale: float = 1.0):
    """
    Draws a reticle on the image at the given (u,v) position
    """
    if alpha < 1.0:
        img = img_raw.copy()
    else:
        img = img_raw
    white = (255, 255, 255)
    cv2.circle(img, (u, v), int(10 * scale), label_color, 1)
    cv2.circle(img, (u, v), int(11 * scale), white, 1)
    cv2.circle(img, (u, v), int(12 * scale), label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), label_color, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), label_color, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), label_color, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), label_color, 1)
    if alpha < 1.0:
        return cv2.addWeighted(img, alpha, img_raw, 1 - alpha, 0)
    else:
        return img


def get_gaussian_kernel_heatmap(norm_diffs: np.ndarray, variance):
    """
    Compute RGB heatmap from norm diffs
    Args:
        norm_diffs (H, W): distances in descriptor space to a given keypoint
        variance: the variance of the kernel

    Returns: RGB image (H, W, 3)

    """
    heatmap = np.exp(-np.copy(norm_diffs) / variance) * 255
    return cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)


def get_crop_resize_param(
        orig_size_hw: Union[tuple, list],
        new_size_hw: Union[tuple, list],
):
    """
    Returns center cropped image
    """
    orig_hw_ratio = orig_size_hw[0] / orig_size_hw[1]
    new_hw_ratio = new_size_hw[0] / new_size_hw[1]

    if orig_hw_ratio > new_hw_ratio:
        # crop the height
        crop_size = (int(orig_size_hw[1]) * new_hw_ratio, orig_size_hw[1])
        crop_idx = (int(abs(orig_size_hw[0] - crop_size[0]) * 0.5),
                    int((orig_size_hw[0] + crop_size[0]) * 0.5),
                    0,
                    orig_size_hw[1])
        scale = new_size_hw[1] / crop_size[1]
    else:
        # crop the width
        crop_size = (orig_size_hw[0], int(orig_size_hw[0] / new_hw_ratio))
        crop_idx = (0,
                    orig_size_hw[0],
                    int(abs(orig_size_hw[1] - crop_size[1]) * 0.5),
                    int((orig_size_hw[1] + crop_size[1]) * 0.5))
        scale = new_size_hw[0] / crop_size[0]

    return crop_size, crop_idx, scale


def crop_and_resize(image: np.ndarray, crop_idx: tuple, target_size: tuple) -> np.ndarray:
    image_crop = image[crop_idx[0]:crop_idx[1], crop_idx[2]:crop_idx[3]]
    return cv2.resize(image_crop, target_size)


def overlay_mask_on_image(rgb_image: np.ndarray, mask_image: np.ndarray, color: np.ndarray = None, rgb_weights: float = 1.0):
    # Overlay the mask onto the RGB image
    if color is None:
        color = np.random.randint(0, 255, 3).astype(np.uint8)
    rgb_img = copy.deepcopy(rgb_image) * mask_image[..., None]
    background_img = np.logical_not(mask_image).astype(np.uint8)[..., None] * color[None, :3].astype(np.uint8)

    # Add color map binary mask image to the overlay
    overlay = cv2.addWeighted(rgb_img, rgb_weights, background_img, 1, 0)
    return overlay


def overlay_masks_on_image(
        rgb_image: np.ndarray,
        masks: Union[np.ndarray, List[np.ndarray]],
        colors: Union[np.ndarray, List[np.ndarray]] = None,
        rgb_weights: float = 0.8
):
    canvas = np.zeros_like(rgb_image)
    n_masks = len(masks) if isinstance(masks, list) else masks.shape[0]
    if colors is None:
        colors = [np.random.randint(0, 255, 3).astype(np.uint8) for _ in range(n_masks)]
    for mask, color in zip(masks, colors):
        # Overlay the mask on the canvas with the random color and transparency
        mask = np.expand_dims(mask, -1) * color.reshape(1, 1, 3)
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        canvas = cv2.addWeighted(
            canvas, 1,
            mask.astype(np.uint8), rgb_weights, 0)

    return cv2.addWeighted(rgb_image, 1, canvas, 1, 0)


def erode_mask(mask: np.ndarray, erode_radius: int = 10):
    kernel = np.ones((erode_radius, erode_radius), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def dilate_mask(mask: np.ndarray, erode_radius: int = 10):
    kernel = np.ones((erode_radius, erode_radius), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


