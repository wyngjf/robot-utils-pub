import numpy as np
from robot_utils.py.utils import load_dict_from_yaml


class CameraIntrinsics(object):
    def __init__(self, cx, cy, fx, fy, width, height):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.width = width
        self.height = height

        self.K = self.get_camera_matrix()

    def get_camera_matrix(self):
        return np.array([
            [self.fx,       0,       self.cx],
            [0,       self.fy,       self.cy],
            [0,             0,             1]])

    @staticmethod
    def from_yaml_file(filename):
        config = load_dict_from_yaml(filename)

        fx = config['camera_matrix']['data'][0]
        cx = config['camera_matrix']['data'][2]

        fy = config['camera_matrix']['data'][4]
        cy = config['camera_matrix']['data'][5]

        width = config['image_width']   # 640
        height = config['image_height'] # 480

        return CameraIntrinsics(cx, cy, fx, fy, width, height)