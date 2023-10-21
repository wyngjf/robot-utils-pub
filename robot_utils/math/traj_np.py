import numpy as np
from typing import Union
import robot_utils.math.manifold.operator_np as op
import robot_utils.math.transformations as t


def get_quat_derivative_traj(quat_traj: np.ndarray, dt: Union[float, np.ndarray]):
    """
    given a quaternion trajectory, compute the quaternion derivative (dq / dt) trajectory

    Args:
        quat_traj: (T, 4)
        dt: (T, ) or float. If a float value is given, it will be repeated

    Returns: (T, 4) quaternion derivative trajectory (w.r.t. each quaternion in quat_traj locally)

    """
    batch = quat_traj.shape[0]
    angular_vel = get_angular_velocity_traj(quat_traj, dt)                      # (batch, 3)
    angular_vel = np.concatenate((np.zeros((batch, 1)), angular_vel), axis=-1)  # (batch, 4)
    dq_dt = 0.5 * t.quaternion_multiply(quat_traj, angular_vel)  # local representation
    return dq_dt


def flip_quaternions(traj):
    batch, dim, time = traj.shape
    traj = traj.transpose((0, 2, 1))
    new_traj = np.zeros_like(traj)
    q = traj[0, 0]
    for i in range(batch):
        if np.dot(q, traj[i, 0]) < 0:
            new_traj[i, 0] = -traj[i, 0]
        else:
            new_traj[i, 0] = traj[i, 0]

    for n in range(batch):
        q = new_traj[n, 0]
        for t in range(time):
            if np.dot(q, traj[n, t]) < 0:
                new_traj[n, t] = -traj[n, t]
            else:
                new_traj[n, t] = traj[n, t]
    return new_traj.transpose((0, 2, 1))


def get_angular_velocity_traj(quat_traj: np.ndarray, dt: Union[float, np.ndarray]):
    """
    given a quaternion trajectory, compute the angular velocity trajectory

    Args:
        quat_traj: (T, 4)
        dt: (T, ) or float. If a float value is given, it will be repeated

    Returns: (T, 3) angular velocity trajectory

    """
    batch = quat_traj.shape[0]
    if isinstance(dt, float):
        dt = np.ones(batch - 1) * dt
    vel_traj = np.zeros((batch, 3))
    for i in range(batch - 1):
        vel_traj[i] = t.angular_velocities_from_quat(quat_traj[i], quat_traj[i+1], dt[i])
    return vel_traj

