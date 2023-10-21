"""
adapted from open3d examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py
"""
import click
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Union, Tuple
from rich.console import Console
from robot_utils.py.filesystem import create_path
console = Console()


class ImageExtractor:
    def __init__(
            self,
            video: Union[str, Path],
            output: Union[str, Path] = None,
            enable_visualization: bool = False,
            time_slice: Tuple[float, float] = None,
            target_frames: int = 20
    ):
        self.video = video = Path(video)
        if not video.is_file():
            console.print(f"[bold red]{video} not exist")
            exit(1)
        console.rule(f"[bold blue]Extracting images from {video}, target {target_frames} frames")

        self.flag_exit = False
        self.flag_play = True
        self.enable_viz = enable_visualization
        self.time_slice = None if time_slice is None else np.array(time_slice)
        self.target_frames = target_frames

        if output is None:
            output = video.parent
        self.output = create_path(output)
        self.path_rgb = create_path(self.output / "rgb", remove_existing=True)
        self.path_depth = create_path(self.output / "depth", remove_existing=True)

        self.reader = o3d.io.AzureKinectMKVReader()
        self.reader.open(str(video))
        if not self.reader.is_opened():
            console.print(f"[red]Unable to open file {video}")
            exit(1)

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

    def run(self):
        console.log("start")
        if self.enable_viz:
            glfw_key_escape = 256
            glfw_key_space = 32
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.register_key_callback(glfw_key_escape, self.escape_callback)
            vis.register_key_callback(glfw_key_space, self.space_callback)

            vis_geometry_added = False
            vis.create_window('reader', 1920, 540)
            console.print("MKV reader initialized. Press [SPACE] to pause/start, [ESC] to exit.")

        vid = cv2.VideoCapture(str(self.video))
        fps = vid.get(cv2.CAP_PROP_FPS)
        n_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.time_slice is not None:
            duration = n_frames / fps
            frame_slice = np.floor(self.time_slice / duration * n_frames).astype(int)
            console.print(f"[green]extracting frames between {frame_slice[0]} and {frame_slice[1]}")
        else:
            frame_slice = np.array([0, n_frames], dtype=int)

        # we have to guarantee the speed doesn't change, the frames should be sampled in equi-distance
        self.target_frames = min(self.target_frames, frame_slice[1]-frame_slice[0])
        step = int(np.floor((frame_slice[1]-frame_slice[0]) / self.target_frames))
        console.log(f"in total {self.target_frames}, extract in every {step} frames")

        metadata = self.reader.get_metadata()
        o3d.io.write_azure_kinect_mkv_metadata(str(self.output / 'intrinsic.json'), metadata)

        idx = 0
        saving_idx = 0
        while not self.reader.is_eof() and not self.flag_exit:
            if self.flag_play:
                rgbd = self.reader.next_frame()
                if rgbd is None:
                    continue

                if self.enable_viz and not vis_geometry_added:
                    vis.add_geometry(rgbd)
                    vis_geometry_added = True

                if frame_slice[0] <= idx <= frame_slice[1]:
                    if saving_idx % step == 0:
                        name_idx = int(saving_idx // step)
                        o3d.io.write_image(str(self.path_rgb/f"{name_idx:>04d}.jpg"), rgbd.color)
                        o3d.io.write_image(str(self.path_depth/f"{name_idx:>04d}.png"), rgbd.depth)
                    saving_idx += 1
                idx += 1

            if self.enable_viz:
                try:
                    vis.update_geometry(rgbd)
                except NameError:
                    pass
                vis.poll_events()
                vis.update_renderer()

        self.reader.close()
        console.rule(f"[blue]extraction finished")


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--video",   "-v", type=str,       default=None,   help="input mkv video")
@click.option("--output",  "-o", type=str,       default=None,   help="working dir for output images")
@click.option("--start",   "-s", type=float,     default=None,   help="start of the time slice")
@click.option("--end",     "-e", type=float,     default=None,   help="end of the time slice")
@click.option("--frames",  "-f", type=int,       default=None,   help="target number of frames")
def main(video, output, start, end, frames):
    reader = ImageExtractor(video, output, time_slice=(start, end), target_frames=frames)
    reader.run()


if __name__ == '__main__':
    main()
