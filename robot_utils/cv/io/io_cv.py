from pathlib import Path
from typing import Union
from robot_utils.py.filesystem import validate_file

import cv2
import numpy as np


def load_image(filename: Union[str, Path]):
    img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    return img


def load_rgb(filename: Union[str, Path], bgr2rgb: bool = False) -> np.ndarray:
    """
    load RGB image
    Args:
        filename:

    Returns: (h, w, c) dtype uint8

    """
    validate_file(filename, throw_error=True)
    img = cv2.imread(str(filename))
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_depth(filename: Union[str, Path], to_meter: bool = True) -> np.ndarray:
    """
    Read depth image and convert to float type in meter
    Args:
        filename:
        to_meter: True to convert to meter

    Returns: (h, w) dtype float in meter or uint16 in milli-meter

    """
    img = cv2.imread(str(filename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img.astype(np.float64) * 0.001 if to_meter else img.astype(np.float64)


def load_depth_to_color(
        filename: Union[str, Path],
        min_depth: float = 0.0,
        max_depth: float = 3000,
        fixed_max_depth: bool = True
):
    """
    Load depth image and convert to a color map
    Args:
        filename:
        min_depth: in milli-meter
        max_depth: in milli-meter
        fixed_max_depth: True, use the max_depth as end of color map; False, use depth_img.max() as max in colormap

    Returns: colored depth image in uint8
    """
    depth = cv2.imread(str(filename), -1)
    depth = depth * (depth < max_depth) * (depth >= min_depth)
    max_color = max_depth if fixed_max_depth else depth.max()
    return cv2.applyColorMap(np.round(depth / (max_color / 255)).astype(np.uint8), cv2.COLORMAP_JET)


def load_disparity(filename: Union[str, Path]) -> np.ndarray:
    """
    Load disparity maps
    Args:
        filename:

    Returns: (h, w) in float

    """
    disparity_map = cv2.imread(str(filename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return np.float32(disparity_map) / 255.


def load_mask(filename: Union[str, Path], as_binary: bool = False) -> np.ndarray:
    """
    Load mask image
    Args:
        filename:
        as_binary: True: boolean type, otherwise uint8 as 1 or 0

    Returns:

    """
    img = cv2.imread(str(filename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img.astype(bool) if as_binary else img


def write_rgb(filename: Union[str, Path], img: np.ndarray, bgr2rgb: bool = False):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(filename), img)


def write_depth(filename: Union[str, Path], depth_map: np.ndarray, to_meter: bool = False):
    scale = 0.001 if to_meter else 1.0
    cv2.imwrite(str(filename), (depth_map * scale).astype(np.uint16))


def write_colorized_depth(
        filename: Union[str, Path],
        depth_map_meter: np.ndarray,
        to_meter: bool = False,
        min_meter: float = None,
        max_meter: float = None
):
    scale = 0.001 if to_meter else 1.0
    normalized_depth_map = depth_map_meter * scale
    if min_meter is None:
        min_meter = normalized_depth_map.min()
    if max_meter is None:
        max_meter = normalized_depth_map.max()
    if min_meter == max_meter:
        normalized_depth_map = normalized_depth_map - min_meter
    else:
        normalized_depth_map = (normalized_depth_map - min_meter) / (max_meter - min_meter)

    # Apply a colormap to the normalized depth map
    colormap = cv2.applyColorMap(np.uint8(normalized_depth_map * 255), cv2.COLORMAP_JET)
    write_rgb(filename, colormap)


def write_binary_mask(filename: Union[str, Path], binary_mask: np.ndarray):
    cv2.imwrite(str(filename), binary_mask.astype(np.uint8) * 255)

