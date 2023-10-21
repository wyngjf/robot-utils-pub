import logging
import shutil
from typing import Union, List, Tuple
from pathlib import Path
from natsort import natsorted


def get_ordered_subdirs(
        dir_name:   Union[str, Path],
        pattern:    List[str] = None,
        ex_pattern: List[str] = None,
        recursive:  bool = False,
        to_str:     bool = False
) -> Union[List[Path], List[str]]:
    """
    Args:
        dir_name: the directory name to look for sub-directories
        pattern: the pattern to follow when filtering the sub-directory name
        ex_pattern: exclude pattern to filter the file name
        recursive: whether to search recursively
        to_str: convert Path objects to str

    Returns:
        A list of ordered subdirectories in the dir_name that fulfills the pattern and excludes the ex_pattern

    """
    dir_name = Path(dir_name)
    if pattern is None:
        pattern = ['']
    if ex_pattern is None:
        ex_pattern = []
    ext = "**/*" if recursive else "*"

    dirs = natsorted(filter(
        lambda x: x.is_dir() and all([p in str(x.name) for p in pattern]) and all([p not in str(x) for p in ex_pattern]),
        dir_name.glob(ext)
    ))
    if to_str:
        return [str(p) for p in dirs]
    else:
        return dirs


def get_ordered_files(
        dir_name:   Union[str, Path],
        pattern:    List[str] = None,
        ex_pattern: List[str] = None,
        recursive:  bool = False,
        to_str:     bool = False
):
    """
    Args:
        dir_name: the directory name to look for sub-directories
        pattern: the pattern to follow when filtering the sub-directory name
        ex_pattern: exclude pattern to filter the file name
        recursive: whether to search recursively
        to_str: convert Path objects to str

    Returns:
        A list of ordered filenames in the dir_name that fulfills the pattern and excludes the ex_pattern

    """
    dir_name = Path(dir_name)
    if pattern is None:
        pattern = ['']
    if ex_pattern is None:
        ex_pattern = []
    ext = "**/*" if recursive else "*"

    files = natsorted(filter(
        lambda x: x.is_file() and all([p in str(x.name) for p in pattern]) and all([p not in str(x.name) for p in ex_pattern]),
        dir_name.glob(ext)
    ))
    return [str(p) for p in files] if to_str else files


def validate_path(
        path: Union[str, Path, None],
        create: bool = False,
        throw_error: bool = False,
        message: str = ""
) -> Tuple[Union[None, Path], bool]:
    if path is None:
        return path, False
    path = Path(path)
    if path.exists():
        return path, True

    if create:
        logging.info(f"creating path: {path}")
        return create_path(path), True

    logging.error(f"{path} does not exit. {message}")
    if throw_error:
        raise FileNotFoundError

    return path, False


def get_validate_path(
        path: Union[str, Path, None],
        create: bool = False
):
    return validate_path(path, create, True)[0]


def validate_file(
        filename: Union[str, Path],
        throw_error: bool = False,
        message: str = ""
) -> Tuple[Path, bool]:
    filename = Path(filename)
    if filename.is_file():
        return filename, True

    logging.error(f"{filename} does not exit. {message}")
    if throw_error:
        raise FileNotFoundError
    return filename, False


def get_validate_file(
    filename: Union[str, Path]
):
    return validate_file(filename, throw_error=True)[0]


def create_path(path: Union[str, Path], remove_existing: bool = False, auto: bool = False) -> Path:
    """
    check if dir exist, otherwise create new dir
    Args:
        path: the absolute path to be created. If already exists, don't do anything
        remove_existing: True, remove the existing path, ask user to confirm. False, use the existing path
        auto: if remove_existing is True, whether to automatically (auto=True) remove the folder or ask the user to
            choose (auto=False).

    Returns: Path object of the created folder

    """
    path = Path(path)
    if remove_existing and path.exists():
        logging.warning(f"Attempt to remove {path}?")
        if auto:
            shutil.rmtree(path)
        elif input("delete y/N? ") == 'y':
            shutil.rmtree(path)
        else:
            raise ValueError("code try to remove path, but user terminated")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def delete_path(path: Union[str, Path]):
    path = Path(path)
    if path.is_dir() or path.is_file():
        logging.warning(f"Attempt to remove {path}?")
        if input("delete y/N? ") == 'y':
            shutil.rmtree(path)


def move_path(src: Union[str, Path], dst: Union[str, Path], remove_existing=False):
    src = Path(src)
    if src.is_dir():
        create_path(dst, remove_existing)
    shutil.move(src, dst)


def rename_path(src: Union[str, Path], dst: Union[str, Path]):
    src = Path(src)
    if src.is_dir() or src.is_file():
        src.rename(dst)


def get_home():
    return Path.home()


def copy2(src: Union[str, Path], dst: Union[str, Path]):
    """
    src need to be a file, while dst can be either a directory or the target filename
    """
    shutil.copy2(src, dst)


def copytree(src: Union[str, Path], dst: Union[str, Path], dirs_exist_ok: bool = True):
    shutil.copytree(src, dst, dirs_exist_ok=dirs_exist_ok)
