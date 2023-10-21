import os
import cv2
from pathlib import Path
from multiprocessing import Pool
from typing import Union, Literal, List, Tuple
from functools import partial
from PIL import Image
import numpy as np
from rich.progress import track

from robot_utils import console
from robot_utils.cv.io.io_pil import read_img
from robot_utils.cv.io.io_cv import load_image, load_rgb, load_depth_to_color
from robot_utils.py.filesystem import validate_path, get_ordered_files, create_path
from robot_utils.py.modules import install_package
from robot_utils.py.system import run


def image_to_video(
        img_dir:    Union[str, Path],
        video_file: Union[str, Path] = None,
        mode:       Literal["any", "rgb", "depth", "depth_color", "mask"] = "any",
        pattern:    List[str] = None,
        recursive:  bool = False,
        fps:        int = 30,
        resolution: Tuple[int, int] = None,
        codec:      Literal["XVID", "MPEG", "MJPG", "mp4v"] = "XVID",
        **kwargs
):
    """
    supported codec can be XVID, MPEG, MJPG, mp4v, if you are not sure about the resolution, leave it None.
    """
    img_dir, _ = validate_path(img_dir, throw_error=True)
    if video_file is None:
        video_file = img_dir.parent / f"{img_dir.stem}.mp4"
    console.print(f"[blue]reading images from {img_dir}")

    image_list = get_ordered_files(img_dir, pattern=pattern, recursive=recursive)
    if len(image_list) == 0:
        console.print("[bold red]the folder you are working on doesn't have any image that matches your pattern")
        exit(1)

    load_func = {
        "any": load_image,
        "rgb": load_rgb,
        "depth": load_rgb,
        "mask": load_rgb,
        "depth_color": partial(load_depth_to_color, **kwargs)
    }[mode]

    pool = Pool(os.cpu_count())
    image_list = pool.map(load_func, image_list)
    image_arrays_to_video(image_list, video_file, fps, resolution, codec)


def image_arrays_to_video(
        images:     List[np.ndarray],
        video_file: Union[str, Path] = None,
        fps:        int = 30,
        resolution: Tuple[int, int] = None,
        codec:      Literal["XVID", "MPEG", "MJPG", "mp4v"] = "XVID",
        bgr2rgb:    bool = False,
        **kwargs
):
    if resolution is None:
        shape = images[0].shape
        resolution = (shape[1], shape[0])
        console.print(f"[green]using resolution from image {resolution}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(video_file), fourcc, fps, resolution)

    for img in track(images):
        out.write(img[..., ::-1] if bgr2rgb else img)

    console.print(f"writing video to {video_file}")
    out.release()
    console.log("[bold cyan]done")


def image_to_gif(
        img_dir:    Union[str, Path],
        pattern:    List[str] = None,
        gif_file:   Union[str, Path] = None,
        duration:   float = 150,
        loop:       int = 0,
):
    img_dir, _ = validate_path(img_dir, throw_error=True)
    if gif_file is None:
        gif_file = img_dir.parent / f"{img_dir.stem}.gif"

    pool = Pool(os.cpu_count())
    image_list = get_ordered_files(img_dir, pattern)
    img, *imgs = pool.map(read_img, image_list)
    img.save(fp=gif_file, format='GIF', append_images=imgs, save_all=True, duration=duration, loop=loop)
    console.log("[bold cyan]done")


def image_arrays_to_gif(
        images:     List[np.ndarray],
        gif_file:   Union[str, Path] = None,
        duration:   float = 150,
        loop:       int = 0,
):
    pool = Pool(os.cpu_count())
    pil_images = pool.map(Image.fromarray, images)
    pil_images[0].save(
        fp=gif_file, format='GIF', append_images=pil_images[1:], save_all=True, duration=duration, loop=loop
    )
    console.log("[bold cyan]done")


def video_to_gif(
        video_file:     Union[str, Path],
        gif_file:       Union[str, Path] = None,
        fps:            int = 10,
):
    """
    great blog: http://zulko.github.io/blog/2014/01/23/making-animated-gifs-from-video-files-with-python/
    """
    install_package("moviepy", "moviepy")
    from moviepy.editor import VideoFileClip, CompositeVideoClip  # , TextClip

    video_file = Path(video_file)
    if not video_file.is_file():
        console.print(f"[red bold]video file {video_file} not found")
        exit(1)

    if gif_file is None:
        gif_file = video_file.parent / f"{video_file.stem}.gif"

    clip = VideoFileClip(str(video_file), audio=False).subclip(0, 5)
    # text = (
    #     TextClip("20X", fontsize=50, color='white', font='Amiri-Bold').set_pos((30, 500)).set_duration(clip.duration))

    composition = CompositeVideoClip([clip])
    # composition.write_gif('olaf.gif', fps=10, fuzz=2)
    composition.write_gif(gif_file, fps=fps)
    console.log("[bold cyan]done")


def get_n_frames_in_video(video_file: Union[str, Path]) -> int:
    """
    Args:
        video_file: Path to a video.

    Returns: The number of frames in a video.
    """
    cmd = f"ffprobe -v error -select_streams v:0 -count_packets \
            -show_entries stream=nb_read_packets -of csv=p=0 \"{video_file}\""
    output = run(cmd, capture_output=True)
    if output is None:
        raise RuntimeError("command failed !")
    output = output.strip(" ,\t\n\r")
    return int(output)


def extract_images_from_video(
        video_file: Union[str, Path],
        target_frames: int = 100,
        time_slice: str = "",
):
    """
    Extract a desired number (target_frames) of image frames from a given video within the
    given time slice. Outputs images to 'image_dir' in the same folder
    """
    video_file = Path(video_file)
    if not video_file.is_file():
        console.rule("[red]missing video file, skipping ffmpeg")
        return
    console.rule("[blue]Extracting images from video")
    image_path = create_path(video_file.parent / "images", remove_existing=True)
    image_file = image_path / "%04d.jpg"

    cmd = f"ffmpeg -i {video_file} -qscale:v 1 -qmin 1"

    n_frames = get_n_frames_in_video(video_file)
    step = n_frames // target_frames
    if step < 1:
        raise ValueError(f"Target number of frames {target_frames} cannot be fulfilled")
    cmd += f" -vf thumbnail={step},setpts=N/TB"

    if time_slice:
        start, end = time_slice.split(",")
        cmd += f",select='between(t\,{start}\,{end})'"

    cmd += " -r 1"

    cmd += f" {image_file}"
    run(cmd)


if __name__ == "__main__":
    rgb_path = Path("/media/gao/dataset/kvil/brush/2022-12-29-12-37-25/kvil/shade")
    depth_path = Path("/media/gao/dataset/kvil/brush/2022-12-29-12-37-25/kvil/depth")
    mask_path = Path("/media/gao/dataset/kvil/brush/2022-12-29-12-37-25/kvil/mask")

    image_to_video(rgb_path, video_file=rgb_path.parent / "shade_1.mp4", mode="rgb")
    image_to_video(depth_path, video_file=rgb_path.parent / "depth_0.mp4", mode="depth")
    image_to_video(depth_path, video_file=rgb_path.parent / "depth_1.mp4", mode="depth_color", max_depth=10000)
    image_to_video(mask_path, video_file=rgb_path.parent / "mask_1.mp4", pattern=["visible"], mode="mask")

    image_to_gif(rgb_path, duration=30)
    video_file_name = rgb_path.parent / "shade_1.mp4"
    video_to_gif(video_file_name)
