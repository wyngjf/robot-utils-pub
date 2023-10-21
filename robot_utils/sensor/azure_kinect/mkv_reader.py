"""
python examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input record.mkv --output frames
    http://www.open3d.org/docs/release/tutorial/sensor/azure_kinect.html#open3d-azure-kinect-mkv-reader
"""
import click
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Union

from robot_utils import console
from robot_utils.py.filesystem import create_path, get_ordered_files, get_validate_path, get_validate_file
from robot_utils.serialize.dataclass import load_dict_from_yaml, dump_data_to_yaml
from robot_utils.cv.io.io_cv import write_rgb, write_depth, write_colorized_depth

from robot_vision.dataset.azure_kinect.type import AzureKinectCameraParam


class ReaderWithCallback:
    def __init__(self, mkv_dir, output_path: Union[str, Path], enable_visualization: bool = False):
        self.source_dir = get_validate_path(mkv_dir)
        mkv_file = get_ordered_files(mkv_dir, pattern=[".mkv"], to_str=True)[-1]
        console.log(f"[blue]loading {mkv_file}")

        self.output_path = create_path(output_path)

        self.reader = o3d.io.AzureKinectMKVReader()
        self.reader.open(mkv_file)
        if not self.reader.is_opened():
            raise RuntimeError("Unable to open file {}".format(mkv_file))

        self.flag_resize = False
        self.flag_exit = False
        self.flag_play = True

        self.enable_viz = enable_visualization

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def space_callback(self, vis):
        if self.flag_play:
            print('Playback paused, press [SPACE] to continue.')
        else:
            print('Playback resumed, press [SPACE] to pause.')
        self.flag_play = not self.flag_play
        return False

    def serialize(self):
        '''
        Change intrinsic.json to pdc compatible camera_info.yaml file.
        '''

        # def get_intrinsics(data):
        #     '''
        #     Change the camera intrinsics gotten from other resolution to 640X480,
        #     crop the width of the original image, and remain the height information based on the Azure Kinect sensor property,
        #     others may vary.
        #     '''
        #     w = data['width']
        #     h = data['height']
        #     c_x = data['intrinsic_matrix'][6]
        #     c_y = data['intrinsic_matrix'][7]
        #     f_x = data['intrinsic_matrix'][0]
        #     f_y = data['intrinsic_matrix'][4]
        #
        #     k = h / 480
        #     w_cropped = 640 * k
        #
        #     c_x1 = c_x - (w - w_cropped) / 2
        #
        #     c_x2 = c_x1 / k
        #     c_y2 = c_y / k
        #     f_x2 = f_x / k
        #     f_y2 = f_y / k
        #
        #     return f_x2, c_x2, f_y2, c_y2, 1.0
        #
        # def get_camera_matrix(data):
        #     m = [0.] * 9
        #     m[0], m[2], m[4], m[5], m[8] = get_intrinsics(data)
        #     return m
        #
        # def get_projection_matrix(data):
        #     m = [0] * 12
        #     m[0], m[2], m[5], m[6], m[10] = get_intrinsics(data)
        #     return m
        #
        # def get_rectification_matrix():
        #     m = [0.] * 9
        #     m[0], m[4], m[8] = 1.0, 1.0, 1.0
        #     return m
        #
        # def get_intrinsic_matrix_json(data):
        #     m = [0.] * 9
        #     m[0], m[6], m[4], m[7], m[8] = get_intrinsics(data)
        #     return m

        # read json file
        # data = load_dict_from_yaml(get_validate_file(self.source_dir / 'intrinsic.json'))
        with open(self.source_dir / 'intrinsic.json', 'r') as stream:
            try:
                data = json.load(stream)
            except json.JSONDecodeError as exc:
                print(exc)

        assert bool(data) != False

        d = AzureKinectCameraParam()
        d.image_size.extend([data["height"], data["width"]])
        d.intrinsic_matrix = np.array(data["intrinsic_matrix"])
        dump_data_to_yaml(AzureKinectCameraParam, d, self.output_path / "param.yaml")
        # d = dict()
        #
        # d['camera_matrix'] = dict()
        # d['camera_matrix']['cols'] = 3
        # d['camera_matrix']['rows'] = 3
        # d['camera_matrix']['data'] = get_camera_matrix(data)
        #
        # # TODO
        # # d['distortion_model']               = msg.distortion_model
        # # d['distortion_coefficients']        = dict()
        # # d['distortion_coefficients']['cols']= 5
        # # d['distortion_coefficients']['rows']= 1
        # # d['distortion_coefficients']['data']= list(msg.D)
        #
        # d['projection_matrix'] = dict()
        # d['projection_matrix']['cols'] = 4
        # d['projection_matrix']['rows'] = 3
        # d['projection_matrix']['data'] = get_projection_matrix(data)
        #
        # d['rectification_matrix'] = dict()
        # d['rectification_matrix']['cols'] = 3
        # d['rectification_matrix']['rows'] = 3
        # d['rectification_matrix']['data'] = get_rectification_matrix()
        #
        # d['serial_number'] = data['serial_number']
        # d['stream_length_usec'] = data['stream_length_usec']
        # d['color_mode'] = data['color_mode']
        # d['depth_mode'] = data['depth_mode']
        # d['image_width'] = 640
        # d['image_height'] = 480

        # # save .yaml  file
        # with open('{}/camera_info.yaml'.format(self.output), 'w') as outfile:
        #     yaml.dump(d, outfile, default_flow_style=False)
        #
        # e = dict()
        #
        # e['width'] = 640
        # e['height'] = 480
        #
        # e['intrinsic_matrix'] = get_intrinsic_matrix_json(data)
        #
        # # save .json file
        # with open('{}/camera_azurekinect.json'.format(self.output), 'w') as outfile:
        #     json.dump(e, outfile)

    # def initialize_resize(self, img):
    #     '''
    #     Change other resolution image to 640X480.
    #     First crop the width,
    #     then resize to the desire size.
    #     (method based on the Azure Kinect, others may vary)
    #
    #     https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.CenterCrop
    #     '''
    #     scale = (480, 640)
    #     k = img.shape[0] / 480
    #     w_cropped = k * 640
    #
    #     self.resize_img = transforms.Compose([
    #         transforms.CenterCrop((img.shape[0], w_cropped)),
    #         transforms.Resize(scale, Image.BILINEAR),
    #     ])
    #
    #     self.resize_depth = transforms.Compose([
    #         transforms.CenterCrop((img.shape[0], w_cropped)),
    #         transforms.Resize(scale, Image.NEAREST),
    #     ])
    #
    #     self.crop_img = transforms.Compose([
    #         transforms.CenterCrop((480, 640)),
    #     ])
    #
    # def resize(self, img, type, filename):
    #     if not self.flag_resize:
    #         self.initialize_resize(img)
    #         self.flag_resize = True
    #
    #     if type == 'color':
    #         img = self.resize_img(Image.fromarray(img))
    #         img.save(filename)
    #         # img = np.array(img)
    #     elif type == 'depth':
    #         img = self.resize_depth(Image.fromarray(img))
    #         img.save(filename)
    #         # img = np.array(img)
    #     elif type == 'crop':
    #         img = self.crop_img(Image.fromarray(img))
    #         img.save(filename)
    #     else:
    #         raise Exception

    def run(self, rgb_folder_name: str = "images", depth_folder_name: str = "depth"):
        if self.enable_viz:
            glfw_key_escape = 256
            glfw_key_space = 32
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.register_key_callback(glfw_key_escape, self.escape_callback)
            vis.register_key_callback(glfw_key_space, self.space_callback)

            vis_geometry_added = False
            vis.create_window('reader', 1920, 540)

        if not (self.source_dir / 'intrinsic.json').is_file():
            metadata = self.reader.get_metadata()
            o3d.io.write_azure_kinect_mkv_metadata(str(self.source_dir / 'intrinsic.json'), metadata)
        self.serialize()

        idx = 0
        rgb_path = create_path(self.output_path / rgb_folder_name)
        depth_path = create_path(self.output_path / depth_folder_name)
        while not self.reader.is_eof() and not self.flag_exit:
            if self.flag_play:
                rgbd = self.reader.next_frame()

                if rgbd is None:
                    continue

                if self.enable_viz and not vis_geometry_added:
                    vis.add_geometry(rgbd)
                    vis_geometry_added = True

                write_rgb(rgb_path / f"{idx:>06d}.jpg", np.asarray(rgbd.color).astype(np.uint8), bgr2rgb=True)
                write_depth(depth_path / f"{idx:>06d}.png", np.asarray(rgbd.depth).astype(np.uint16))
                write_colorized_depth(depth_path / f"v_{idx:>06d}.png", np.asarray(rgbd.depth), min_meter=300, max_meter=5000)

                idx += 1

            if self.enable_viz:
                try:
                    vis.update_geometry(rgbd)
                except NameError:
                    pass

                vis.poll_events()
                vis.update_renderer()

        self.reader.close()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--video",   "-v", type=str,       default=None,   help="input mkv video")
@click.option("--output",  "-o", type=str,       default=None,   help="working dir for output images")
def main(video, output):
    if video is None:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    if output is None:
        print('No output path, only play mkv')
    else:
        try:
            create_path(output)
        except (PermissionError, FileExistsError):
            print('Unable to mkdir {}, only play mkv'.format(output))
            output = None

    reader = ReaderWithCallback(video, output)
    reader.run()


if __name__ == '__main__':
    main()
