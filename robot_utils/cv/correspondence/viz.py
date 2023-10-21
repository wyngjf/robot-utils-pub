from copy import deepcopy
import cv2
import numpy as np


def viz_points_with_color_map(
        image: np.ndarray,          # (h, w, 3)
        uv: np.ndarray,             # (N, 2)
        colors: np.ndarray = None,  # (N, 3)
        copy_image: bool = True
) -> np.ndarray:
    img = deepcopy(image) if copy_image else image

    if colors is None:
        colors = np.ones((uv.shape[0], 3)) * np.random.randint(0, 255, (1, 3))

    colors = colors.astype(int)
    for _uv, _color in zip(uv, colors):
        cv2.circle(img, (_uv[0], _uv[1]), 4, _color[:3].tolist(), 1)

    return img

