from pathlib import Path
from typing import List, Any, Literal, Dict

from distlib.database import DistributionPath
from marshmallow_dataclass import dataclass

from robot_utils import console
from robot_utils.pkg.pkg_dep_graph import get_install_order
from robot_utils.py.filesystem import validate_path, get_ordered_files, create_path
from robot_utils.py.interact import ask_checkbox
from robot_utils.py.system import run

from robot_utils.py.utils import default_field, load_dataclass, dump_data_to_yaml


def get_installed_packages():
    import logging
    logging.disable(logging.CRITICAL)
    dist_path = DistributionPath(include_egg=True)
    packages = [d.name for d in dist_path.get_distributions()]
    logging.disable(logging.NOTSET)
    return packages


@dataclass
class GitConfig:
    url:            str = ""
    commit:         str = None
    recursive:      bool = False
    submodule:      List[str] = default_field([])
    as_submodule:   str = None
    installed:      bool = False

    def run(self, name: str, path: Path, parent_path: Path):
        proj_dir = path / name
        if self.as_submodule:
            console.log(f"[bold green]updating submodule: {self.as_submodule}")
            run(f"git submodule update --init --remote {self.as_submodule}", working_dir=str(parent_path))

        if self.installed or proj_dir.exists():
            validate_path(proj_dir, throw_error=True)
            if not self.commit:
                run("git pull", working_dir=str(proj_dir))
        else:
            if self.url:
                recursive = " --recursive " if self.recursive else ""
                run(f"git clone {recursive} {self.url} {name}", working_dir=str(path))
                validate_path(proj_dir, throw_error=True)
            if self.commit:
                run(f"git checkout {self.commit}", working_dir=str(proj_dir))

            if self.submodule:
                for sub_mod in self.submodule:
                    console.log(f"[bold green]updating submodule: {sub_mod}")
                    run(f"git submodule update --init --remote {sub_mod}", working_dir=str(proj_dir))


@dataclass
class UnPack:
    cmd:        str = ""


@dataclass
class SymLink:
    src_rel:    str = ""
    target_rel: str = ""

    def run(self, src_dir: Path, target_dir: Path):
        if not self.src_rel or not self.target_rel:
            raise ValueError(f"the configuration of the symlink for src {self.src_rel} "
                             f"or target {self.target_rel} is not specified")
        src = src_dir / self.src_rel
        target = target_dir / self.target_rel

        if target.is_symlink():
            target.unlink(missing_ok=True)
        console.log(f"link {target} to {src}")
        create_path(target.parent)
        target.symlink_to(src)


@dataclass
class WgetConfig:
    rel_path:   str = None
    name:       str = None
    url:        str = None
    background: bool = True
    unpack:     str = None
    sym:        SymLink = None

    def run(self, path_prefix: Path, proj_dir: Path = None):
        location = create_path(path_prefix / self.rel_path)
        cmd = f"wget -P {str(location)} "
        if not self.name:
            self.name = Path(self.url).name
        filename = location / self.name
        if not filename.is_file():
            cmd += f" -O {self.name} "
            if self.background:
                cmd += " -b "
            cmd += f" {self.url} "
            create_path(location)
            console.log(f"[yellow]Downloading\n{cmd}")
            run(cmd, working_dir=str(location))
        else:
            console.log(f"[bold red]{self.name} exists, skip downloading")

        if self.unpack:
            stem_name = filename.name.split(".")[0]
            if filename.is_file() and not (location / stem_name).is_dir():
                console.log(f"unpack {filename} with [bold cyan]{self.unpack}")
                run(self.unpack, working_dir=str(location))

        if self.sym is not None:
            self.sym.run(location, proj_dir)


@dataclass
class GDownConfig:
    rel_path:   str = None
    name:       str = None
    url:        str = None
    background: bool = False
    md5:        str = None
    extract:    bool = False
    sym:        SymLink = None

    def run(self, path_prefix: Path, proj_dir: Path = None):
        if not self.name:
            raise ValueError(f"you have to specify the name of the file")
        import gdown

        location = create_path(path_prefix / self.rel_path)
        if not (location / self.name).is_file():
            gdown.download(
                url=self.url,
                output=str(location / self.name),
                quiet=self.background,
                fuzzy=True,
            )
        else:
            console.log(f"[bold red]{self.name} exists, skip downloading")

        if self.sym is not None:
            self.sym.run(location, proj_dir)


@dataclass
class DataConfig:
    wget:   List[WgetConfig] = None
    gdown:  List[GDownConfig] = None


@dataclass
class PIP:
    deps:                   List[str] = default_field([])
    ignore_installed:       bool = False
    force_reinstall:        bool = False
    this_pkg_install_mode:  Literal['develop', 'release', 'custom', 'None'] = 'None'
    custom_cmd:             List[str] = default_field([])
    features:               List[str] = default_field([])

    def run(self, proj_dir: Path = None):
        add_cfg = " --ignore-installed " if self.ignore_installed else ""
        add_cfg += " --force_reinstall " if self.force_reinstall else ""
        feature_cmd = f"[{','.join(self.features)}]" if self.features else ""

        if len(self.deps) > 0:
            pip_cmd = f"pip install {' '.join(self.deps)} {add_cfg}"
            console.log(f"install deps with \n[bold cyan]{pip_cmd}")
            run(pip_cmd)

        this_pkg_install_cmd = self._validate_this_pkg_install_mode()
        if this_pkg_install_cmd is not None:
            this_pkg_install_cmd = f"{this_pkg_install_cmd}{feature_cmd} {add_cfg}"
            console.log(f"install this package with \n[bold cyan]{this_pkg_install_cmd}")
            run(this_pkg_install_cmd, working_dir=str(proj_dir))

        if len(self.custom_cmd) > 0:
            for cmd in self.custom_cmd:
                console.log(f"run custom command \n[bold cyan]{cmd}")
                run(cmd, working_dir=str(proj_dir))

    def _validate_this_pkg_install_mode(self):
        if self.this_pkg_install_mode == "develop":
            return f"pip install -e ."
        elif self.this_pkg_install_mode == "release":
            return f"pip install ."
        else:
            return None


@dataclass
class InstallConfig:
    git:                GitConfig = None
    pip:                PIP = None
    sys:                List[str] = default_field([])
    cmake:              List[Any] = default_field([])
    depend_on:          List[str] = default_field([])
    data:               DataConfig = None
    custom_cmd_deps:    List[str] = default_field([])
    custom_cmd_parent:  List[str] = default_field([])

    def run(self, name: str, path: Path, data_path: Path = None, parent_package_path: Path = None):
        console.rule(f"[bold cyan]Installing {name}")
        proj_dir = path / name

        if self.git is not None:
            self.git.run(name, path, parent_package_path)

        if self.pip is not None:
            self.pip.run(proj_dir)

        if len(self.custom_cmd_parent) > 0:
            for step in self.custom_cmd_parent:
                run(step, working_dir=str(parent_package_path))

        if len(self.custom_cmd_deps) > 0:
            for step in self.custom_cmd_deps:
                run(step, working_dir=str(proj_dir))

        if self.data is not None:
            if data_path is None:
                console.log(f"[bold red]User configured data download but no data path is provided")
                raise ValueError()
            if self.data.wget is not None and len(self.data.wget) > 0:
                for w in self.data.wget:
                    w.run(data_path, proj_dir)
            if self.data.gdown is not None and len(self.data.gdown) > 0:
                for w in self.data.gdown:
                    w.run(data_path, proj_dir)

        console.rule(f"[bold green]{name} is installed to {proj_dir}")


def create_deps_template(filename: Path):
    dump_data_to_yaml(InstallConfig, InstallConfig(), filename)


def install_deps(parent_package_path: Path, config_path: Path, install_path: Path, data_path: Path = None):
    pkg_cfg = get_ordered_files(config_path, pattern=[".yaml"])
    pkg_name = [p.stem for p in pkg_cfg]

    console.rule()
    selected_pkg = ask_checkbox("Select packages to install (space to select, arrow to navigate) ", pkg_name)
    idx = [pkg_name.index(p) for p in selected_pkg]

    installed_pkg = get_installed_packages()

    pkg_list = []
    pkg_cfg_dict: Dict[str, InstallConfig] = {}
    for i in idx:
        name = pkg_name[i]
        c = load_dataclass(InstallConfig, pkg_cfg[i])
        c.installed = name in installed_pkg
        pkg_cfg_dict[name] = c
        pkg_list.append((name, c.depend_on))

    install_order = get_install_order(pkg_list)
    console.log(f"[bold green]Installing order based on dependency chain:\n"
                f"[bold yellow]{' --> '.join(install_order)}")
    for pkg in install_order:
        if pkg not in pkg_cfg_dict:
            pkg_cfg_dict[pkg] = load_dataclass(InstallConfig, pkg_cfg[pkg_name.index(pkg)])
        pkg_cfg_dict[pkg].run(pkg, install_path, data_path, parent_package_path)
