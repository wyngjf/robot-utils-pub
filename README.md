# robot-utils
robot utils include utility functions for robotic control and computer vision development in Python.

## Installation
This package is tested with `Python >=3.8` (tested on `3.8, 3.9, 3.10`) and the latest `PyTorch 1.13.0` (as well as version `>1.8`)

If you have conda installed, create a conda virtual environment with name `venv`. 
```
# e.g. with Python 3.8
conda create -n venv python=3.8
```
You can also use other virtual environments as you like.

Then clone this package and install as follows.
```
git clone https://gitlab.com/jianfenggaobit/robot-utils.git
# or for H2T users
git clone git@git.h2t.iar.kit.edu:sw/machine-learning-control/robot-utils.git

cd robot-utils
pip install -e .
```

## Run teleoperation mode

Setup HTC Vive on your computer and launch SteamVR. With both controllers being recognized on the lab pc, you can run this on the lab pc:

```
python -m robot-utils.armar.teleoperation.holistic_teleop
```

The python script is straightforwared. It uses [triad_openvr](https://github.com/TriadSemi/triad_openvr), which is a wrapper around the openvr bindings, and connects them to these Python bindings that instanciate the low-level controllers. They work well for HTC Vive, but I recommend switching to OpenXR and ALVR in the future.

[This issue in Admin Organization is an explaination on how to setup SteamVR on Lab PCs.](https://git.h2t.iar.kit.edu/orga/admin-organization/-/issues/337)


**PyTorch**

If you need this package for machine learning study, please refer to the 
[official instructions](https://pytorch.org/get-started/locally/)  to install the latest PyTorch 1.13.0.
