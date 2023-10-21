import os
import sys
import subprocess
import logging
from importlib.util import find_spec
from importlib import import_module


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_and_import(module_name, package_name):
    pkg_found = find_spec(module_name) is not None
    if pkg_found:
        return import_module(module_name)
    else:
        logging.warning(f"{module_name} not found, would you like me to install {package_name} for you?")
        answer = input("Proceed y/n? ")
        if answer == 'y':
            try:
                install(package_name)
                logging.info(f"{package_name} installed")
                return import_module(module_name)
            except ImportError:
                logging.warning(f"Something happened! you have to manual install {package_name} and retry")
        else:
            logging.warning(f"you have to manually install {package_name}")
            exit()


def install_package(module_name, package_name):
    pkg_found = find_spec(module_name) is not None
    if not pkg_found:
        logging.warning(f"{module_name} not found, would you like me to install {package_name} for you?")
        answer = input("Proceed y/n? ")
        if answer == 'y':
            try:
                install(package_name)
                logging.info(f"{package_name} installed")
                return find_spec(module_name) is not None
            except ImportError:
                logging.warning(f"Something happened! you have to manual install {package_name} and retry")
        else:
            return pkg_found


def add_path_from_env_variable(env_variable: str):
    module_path = os.environ.get(env_variable)
    if not module_path:
        raise EnvironmentError(f"missing env variable {env_variable}")
    sys.path.append(module_path)
