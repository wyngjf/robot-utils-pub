import os
import yaml
import click
import datetime
import open3d as o3d
from robot_utils.py.filesystem import create_path


class RecorderWithCallback:
    """
    adapted from examples/python/reconstruction_system/sensors/azure_kinect_recorder.py
    see also http://www.open3d.org/docs/release/tutorial/sensor/azure_kinect.html#open3d-azure-kinect-recorder
    """
    def __init__(
            self,
            azure_config: str,
            recording_path: str,
            folder_name: str,
            obj: str = "OBJECT",
            info: str = "",
            align_depth_to_color: bool = False,
            sensor_index: int = 0
    ):
        # initialize sensor
        if azure_config is not None:
            self.config = o3d.io.read_azure_kinect_sensor_config(azure_config)
        else:
            self.config = o3d.io.AzureKinectSensorConfig()
        self.recorder = o3d.io.AzureKinectRecorder(self.config, sensor_index)
        if not self.recorder.init_sensor():
            raise RuntimeError('Failed to connect to sensor')

        self.flag_exit = False
        self.flag_record = False
        self.raw_dir = os.path.join(recording_path, folder_name)
        self.info_dir = os.path.join(self.raw_dir, 'info.yaml')
        self.filename = os.path.join(self.raw_dir, f'{folder_name}.mkv')
        self.align_depth_to_color = align_depth_to_color

        self.info = {
            'object_id': obj,
            'folder_name': folder_name,
            'record_info': info,
        }

    def escape_callback(self, vis):
        self.flag_exit = True
        if self.recorder.is_record_created():
            with open(self.info_dir, 'w') as outfile:
                yaml.dump(self.info, outfile, default_flow_style=False)
            print('Recording finished.')
        else:
            print('Nothing has been recorded.')
        return False

    def space_callback(self, vis):
        create_path(self.raw_dir)
        if self.flag_record:
            print('Recording paused.\nPress [Space] to continue.\nPress [ESC] to save and exit.\n')
            self.flag_record = False

        elif not self.recorder.is_record_created():
            if self.recorder.open_record(self.filename):
                print('Recording started.\nPress [SPACE] to pause.\nPress [ESC] to save and exit.\n')
                self.flag_record = True

        else:
            print('Recording resumed, video may be discontinuous.\nPress [SPACE] to pause.\nPress [ESC] to save and exit.\n')
            self.flag_record = True

        return False

    def run(self):
        glfw_key_escape = 256
        glfw_key_space = 32

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.register_key_callback(glfw_key_space, self.space_callback)
        vis.create_window('recorder', 1920, 540)

        print("Recorder initialized. Press [SPACE] to start.\nPress [ESC] to save and exit.")

        vis_geometry_added = False

        while not self.flag_exit:
            rgbd = self.recorder.record_frame(self.flag_record, self.align_depth_to_color)
            if rgbd is None:
                continue

            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            vis.update_geometry(rgbd)
            vis.poll_events()
            vis.update_renderer()

        self.recorder.close_record()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--config",   "-c", type=str,       default=None,   help="input yaml config file for azure kinect")
@click.option("--output",   "-o", type=str,       default=None,   help="output filename in mkv")
@click.option("--device",   "-d", type=int,       default=0,      help="index of sensor")
@click.option("--show",     "-s", is_flag=True,                   help="list available azure kinect sensor index")
def main(config, output, device, show):
    if show:
        print("available azure kinect devices: ")
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if output is None:
        output = '{date:%Y-%m-%d-%H-%M-%S}.mkv'.format(date=datetime.datetime.now())

    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    r = RecorderWithCallback(
        azure_config=config,
        recording_path="./recording",
        folder_name=output,
        sensor_index=device
    )
    r.run()


if __name__ == '__main__':
    main()
