from pathlib import Path
from typing import Union, Literal

from robot_utils.py.filesystem import create_path
from robot_utils.py.system import run
from rich.console import Console


def run_colmap(
    image_dir:       Union[str, Path],
    matcher:         Literal["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"] = "sequential",
    camera_model:    Literal["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"] = "OPENCV",
    camera_params:   str = "",
    vocab_path:      str = ""
):
    console = Console()
    image_dir = Path(image_dir)
    if not image_dir.exists():
        console.rule("[bold red]images not found")
        return

    scene = image_dir.parent
    database = scene / "colmap/colmap.db"
    text = create_path(scene / "colmap/colmap_text", remove_existing=True)
    sparse = create_path(scene / "colmap/colmap_sparse", remove_existing=True)
    console.rule("[bold blue] Running Colmap")
    console.print(f"fetching images from: {image_dir}")
    console.print(f"writing to: {scene / 'colmap'}")

    cmd = f"colmap feature_extractor --ImageReader.camera_model {camera_model}"
    cmd += f" --ImageReader.camera_params \"{camera_params}\" --SiftExtraction.estimate_affine_shape=true"
    cmd += f" --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {database}"
    cmd += f" --image_path {image_dir}"
    run(cmd)

    match_cmd = f"colmap {matcher}_matcher --SiftMatching.guided_matching=true --database_path {database}"
    if vocab_path:
        match_cmd += f" --VocabTreeMatching.vocab_tree_path {vocab_path}"
    run(match_cmd)

    run(f"colmap mapper --database_path {database} --image_path {image_dir} --output_path {sparse}")
    run(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
    run(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")
    console.rule(f"[blue]colmap finished")
