from pathlib import Path
from typing import Union

from PIL import Image


def read_img(filename: Union[str, Path]):
    with Image.open(filename) as temp:
        return temp.copy()


def get_rgb_image(filename: Union[str, Path]):
    with Image.open(filename) as img:
        return img.convert("RGB")
