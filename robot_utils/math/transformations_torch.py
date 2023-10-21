"""
transformations implemented for PyTorch
TODO follow pytorch3d, optimize each function for batch operation
"""
import math
import torch
import numpy as np
import numpy as numpy
from typing import Union, List, Optional, Dict


def identity_matrix():
    """Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> np.allclose(I, np.dot(I, I))
    True
    >>> np.sum(I), np.trace(I)
    (4.0, 4.0)
    >>> np.allclose(I, np.identity(4))
    True

    """
    return np.identity(4)


def translation_matrix(direction):
    """Return matrix to translate by direction vector.

    >>> v = np.random.random(3) - 0.5
    >>> np.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M


def homogeneous_matrix(rot_mat):
    """
    Return homogeneous matrix to from a rotation matrix.
    """
    M = np.identity(4)
    M[:3, :3] = rot_mat
    return M


def translation_from_matrix(matrix):
    """Return translation vector from translation matrix.

    >>> v0 = np.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> np.allclose(v0, v1)
    True

    """
    return np.array(matrix, copy=False)[:3, 3].copy()


def reflection_matrix(point, normal):
    """Return matrix to mirror at plane defined by point and normal vector.

    >>> v0 = np.random.random(4) - 0.5
    >>> v0[3] = 1.
    >>> v1 = np.random.random(3) - 0.5
    >>> R = reflection_matrix(v0, v1)
    >>> np.allclose(2, np.trace(R))
    True
    >>> np.allclose(v0, np.dot(R, v0))
    True
    >>> v2 = v0.copy()
    >>> v2[:3] += v1
    >>> v3 = v0.copy()
    >>> v2[:3] -= v1
    >>> np.allclose(v2, np.dot(R, v3))
    True

    """
    normal = unit_vector(normal[:3])
    M = np.identity(4)
    M[:3, :3] -= 2.0 * np.outer(normal, normal)
    M[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal
    return M


def reflection_from_matrix(matrix):
    """Return mirror plane point and normal vector from reflection matrix.

    >>> v0 = np.random.random(3) - 0.5
    >>> v1 = np.random.random(3) - 0.5
    >>> M0 = reflection_matrix(v0, v1)
    >>> point, normal = reflection_from_matrix(M0)
    >>> M1 = reflection_matrix(point, normal)
    >>> is_same_transform(M0, M1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    # normal: unit eigenvector corresponding to eigenvalue -1
    w, V = np.linalg.eig(M[:3, :3])
    i = np.where(abs(np.real(w) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue -1')
    normal = np.real(V[:, i[0]]).squeeze()
    # point: any unit eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return point, normal


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def scale_matrix(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    >>> v = (np.random.rand(4, 5) - 0.5) * 20
    >>> v[3] = 1
    >>> S = scale_matrix(-1.234)
    >>> np.allclose(np.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)

    """
    if direction is None:
        # uniform scaling
        M = np.diag([factor, factor, factor, 1.0])
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # nonuniform scaling
        direction = unit_vector(direction[:3])
        factor = 1.0 - factor
        M = np.identity(4)
        M[:3, :3] -= factor * np.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = (factor * np.dot(origin[:3], direction)) * direction
    return M


def scale_from_matrix(matrix):
    """Return scaling factor, origin and direction from scaling matrix.

    >>> factor = random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S0 = scale_matrix(factor, origin)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True
    >>> S0 = scale_matrix(factor, origin, direct)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    factor = np.trace(M33) - 2.0
    try:
        # direction: unit eigenvector corresponding to eigenvalue factor
        w, V = np.linalg.eig(M33)
        i = np.where(abs(np.real(w) - factor) < 1e-8)[0][0]
        direction = np.real(V[:, i]).squeeze()
        direction /= vector_norm(direction)
    except IndexError:
        # uniform scaling
        factor = (factor + 2.0) / 3.0
        direction = None
    # origin: any eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no eigenvector corresponding to eigenvalue 1')
    origin = np.real(V[:, i[-1]]).squeeze()
    origin /= origin[3]
    return factor, origin, direction


def projection_matrix(point, normal, direction=None,
                      perspective=None, pseudo=False):
    """Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection direction, or none of both.

    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).

    >>> P = projection_matrix([0, 0, 0], [1, 0, 0])
    >>> np.allclose(P[1:, 1:], np.identity(4)[1:, 1:])
    True
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> P1 = projection_matrix(point, normal, direction=direct)
    >>> P2 = projection_matrix(point, normal, perspective=persp)
    >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> is_same_transform(P2, np.dot(P0, P3))
    True
    >>> P = projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
    >>> v0 = (np.random.rand(4, 5) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(P, v0)
    >>> np.allclose(v1[1], v0[1])
    True
    >>> np.allclose(v1[0], 3-v1[1])
    True

    """
    M = np.identity(4)
    point = np.array(point[:3], dtype=np.float64, copy=False)
    normal = unit_vector(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = np.array(perspective[:3], dtype=np.float64,
                                  copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective-point, normal)
        M[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            M[:3, :3] -= np.outer(normal, normal)
            M[:3, 3] = np.dot(point, normal) * (perspective+normal)
        else:
            M[:3, 3] = np.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = np.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = np.array(direction[:3], dtype=np.float64, copy=False)
        scale = np.dot(direction, normal)
        M[:3, :3] -= np.outer(direction, normal) / scale
        M[:3, 3] = direction * (np.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= np.outer(normal, normal)
        M[:3, 3] = np.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix, pseudo=False):
    """Return projection plane and perspective point from projection matrix.

    Return values are same as arguments for projection_matrix function:
    point, normal, direction, perspective, and pseudo.

    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, direct)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    >>> result = projection_from_matrix(P0, pseudo=False)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> result = projection_from_matrix(P0, pseudo=True)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not pseudo and len(i):
        # point: any eigenvector corresponding to eigenvalue 1
        point = np.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        # direction: unit eigenvector corresponding to eigenvalue 0
        w, V = np.linalg.eig(M33)
        i = np.where(abs(np.real(w)) < 1e-8)[0]
        if not len(i):
            raise ValueError('no eigenvector corresponding to eigenvalue 0')
        direction = np.real(V[:, i[0]]).squeeze()
        direction /= vector_norm(direction)
        # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
        w, V = np.linalg.eig(M33.T)
        i = np.where(abs(np.real(w)) < 1e-8)[0]
        if len(i):
            # parallel projection
            normal = np.real(V[:, i[0]]).squeeze()
            normal /= vector_norm(normal)
            return point, normal, direction, None, False
        else:
            # orthogonal projection, where normal equals direction vector
            return point, direction, None, None, False
    else:
        # perspective projection
        i = np.where(abs(np.real(w)) > 1e-8)[0]
        if not len(i):
            raise ValueError(
                'no eigenvector not corresponding to eigenvalue 0')
        point = np.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        normal = - M[3, :3]
        perspective = M[:3, 3] / np.dot(point[:3], normal)
        if pseudo:
            perspective -= normal
        return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """Return matrix to obtain normalized device coordinates from frustum.

    The frustum bounds are axis-aligned along x (left, right),
    y (bottom, top) and z (near, far).

    Normalized device coordinates are in range [-1, 1] if coordinates are
    inside the frustum.

    If perspective is True the frustum is a truncated pyramid with the
    perspective point at origin and direction along z axis, otherwise an
    orthographic canonical view volume (a box).

    Homogeneous coordinates transformed by the perspective clip matrix
    need to be dehomogenized (divided by w coordinate).

    >>> frustum = np.random.rand(6)
    >>> frustum[1] += frustum[0]
    >>> frustum[3] += frustum[2]
    >>> frustum[5] += frustum[4]
    >>> M = clip_matrix(perspective=False, *frustum)
    >>> np.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    array([-1., -1., -1.,  1.])
    >>> np.dot(M, [frustum[1], frustum[3], frustum[5], 1])
    array([ 1.,  1.,  1.,  1.])
    >>> M = clip_matrix(perspective=True, *frustum)
    >>> v = np.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    >>> v / v[3]
    array([-1., -1., -1.,  1.])
    >>> v = np.dot(M, [frustum[1], frustum[3], frustum[4], 1])
    >>> v / v[3]
    array([ 1.,  1., -1.,  1.])

    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError('invalid frustum')
    if perspective:
        if near <= _EPS:
            raise ValueError('invalid frustum: near <= 0')
        t = 2.0 * near
        M = [[t/(left-right), 0.0, (right+left)/(right-left), 0.0],
             [0.0, t/(bottom-top), (top+bottom)/(top-bottom), 0.0],
             [0.0, 0.0, (far+near)/(near-far), t*far/(far-near)],
             [0.0, 0.0, -1.0, 0.0]]
    else:
        M = [[2.0/(right-left), 0.0, 0.0, (right+left)/(left-right)],
             [0.0, 2.0/(top-bottom), 0.0, (top+bottom)/(bottom-top)],
             [0.0, 0.0, 2.0/(far-near), (far+near)/(near-far)],
             [0.0, 0.0, 0.0, 1.0]]
    return np.array(M)


def shear_matrix(angle, direction, point, normal):
    """Return matrix to shear by angle along direction vector on shear plane.

    The shear plane is defined by a point and normal vector. The direction
    vector must be orthogonal to the plane's normal vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the direction vector and its extent is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S = shear_matrix(angle, direct, point, normal)
    >>> np.allclose(1, np.linalg.det(S))
    True

    """
    normal = unit_vector(normal[:3])
    direction = unit_vector(direction[:3])
    if abs(np.dot(normal, direction)) > 1e-6:
        raise ValueError('direction and normal vectors are not orthogonal')
    angle = math.tan(angle)
    M = np.identity(4)
    M[:3, :3] += angle * np.outer(direction, normal)
    M[:3, 3] = -angle * np.dot(point[:3], normal) * direction
    return M


def shear_from_matrix(matrix):
    """Return shear angle, direction and plane from shear matrix.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S0 = shear_matrix(angle, direct, point, normal)
    >>> angle, direct, point, normal = shear_from_matrix(S0)
    >>> S1 = shear_matrix(angle, direct, point, normal)
    >>> is_same_transform(S0, S1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    w, V = np.linalg.eig(M33)
    i = np.where(abs(np.real(w) - 1.0) < 1e-4)[0]
    if len(i) < 2:
        raise ValueError('no two linear independent eigenvectors found %s' % w)
    V = np.real(V[:, i]).squeeze().T
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = np.cross(V[i0], V[i1])
        w = vector_norm(n)
        if w > lenorm:
            lenorm = w
            normal = n
    normal /= lenorm
    # direction and angle
    direction = np.dot(M33 - np.identity(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    # point: eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError('no eigenvector corresponding to eigenvalue 1')
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return angle, direction, point, normal


def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix([1, 2, 3])
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> np.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> np.allclose(R0, R1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError('M[3, 3] is zero')
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not np.linalg.det(P):
        raise ValueError('matrix is singular')

    scale = np.zeros((3, ))
    shear = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = np.array([0.0, 0.0, 0.0, 1.0])

    translate = M[3, :3].copy()
    M[3, :3] = 0.0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        np.negative(scale, scale)
        np.negative(row, row)

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        # angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective


def compose_matrix(scale=None, shear=None, angles=None, translate=None,
                   perspective=None):
    """Return transformation matrix from sequence of transformations.

    This is the inverse of the decompose_matrix function.

    Sequence of transformations:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    >>> scale = np.random.random(3) - 0.5
    >>> shear = np.random.random(3) - 0.5
    >>> angles = (np.random.random(3) - 0.5) * (2*math.pi)
    >>> trans = np.random.random(3) - 0.5
    >>> persp = np.random.random(4) - 0.5
    >>> M0 = compose_matrix(scale, shear, angles, trans, persp)
    >>> result = decompose_matrix(M0)
    >>> M1 = compose_matrix(*result)
    >>> is_same_transform(M0, M1)
    True

    """
    M = np.identity(4)
    if perspective is not None:
        P = np.identity(4)
        P[3, :] = perspective[:4]
        M = np.dot(M, P)
    if translate is not None:
        T = np.identity(4)
        T[:3, 3] = translate[:3]
        M = np.dot(M, T)
    if angles is not None:
        R = euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
        M = np.dot(M, R)
    if shear is not None:
        Z = np.identity(4)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = np.dot(M, Z)
    if scale is not None:
        S = np.identity(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)
    M /= M[3, 3]
    return M


def orthogonalization_matrix(lengths, angles):
    """Return orthogonalization matrix for crystallographic cell coordinates.

    Angles are expected in degrees.

    The de-orthogonalization matrix is the inverse.

    >>> O = orthogonalization_matrix([10, 10, 10], [90, 90, 90])
    >>> np.allclose(O[:3, :3], np.identity(3, float) * 10)
    True
    >>> O = orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    >>> np.allclose(np.sum(O), 43.063229)
    True

    """
    a, b, c = lengths
    angles = np.radians(angles)
    sina, sinb, _ = np.sin(angles)
    cosa, cosb, cosg = np.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return np.array([
        [ a*sinb*math.sqrt(1.0-co*co),  0.0,    0.0, 0.0],
        [-a*sinb*co,                    b*sina, 0.0, 0.0],
        [ a*cosb,                       b*cosa, c,   0.0],
        [ 0.0,                          0.0,    0.0, 1.0]])


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> R = random_rotation_matrix(np.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> np.allclose(v1, np.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError('input arrays are of wrong shape or type')

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    """Return matrix to transform given 3D point set into second point set.

    v0 and v1 are shape (3, \*) or (4, \*) arrays of at least 3 points.

    The parameters scale and usesvd are explained in the more general
    affine_matrix_from_points function.

    The returned matrix is a similarity or Euclidean transformation matrix.
    This function has a fast C implementation in transformations.c.

    >>> v0 = np.random.rand(3, 10)
    >>> M = superimposition_matrix(v0, v0)
    >>> np.allclose(M, np.identity(4))
    True
    >>> R = random_rotation_matrix(np.random.random(3))
    >>> v0 = [[1,0,0], [0,1,0], [0,0,1], [1,1,1]]
    >>> v1 = np.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> S = scale_matrix(random.random())
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> M = concatenate_matrices(T, R, S)
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-9, 300).reshape(3, -1)
    >>> M = superimposition_matrix(v0, v1, scale=True)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> v = np.empty((4, 100, 3))
    >>> v[:, :, 0] = v0
    >>> M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    >>> np.allclose(v1, np.dot(M, v[:, :, 0]))
    True

    """
    v0 = np.array(v0, dtype=np.float64, copy=False)[:3]
    v1 = np.array(v1, dtype=np.float64, copy=False)[:3]
    return affine_matrix_from_points(v0, v1, shear=False,
                                     scale=scale, usesvd=usesvd)


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not np.allclose(R0, R1): print(axes, "failed")

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    >>> np.allclose(q, [0.99810947, 0.06146124, 0, 0])
    True

    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle/2.0) / qlen
    q[0] = math.cos(angle/2.0)
    return q


def quaternion_matrix(quaternion, rot_only=False):
    """Return homogeneous rotation matrix from quaternion. (w, x, y, z)

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    mat = np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])
    if rot_only:
        return mat[:3, :3]
    return mat


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(np.identity(4), True)
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
    >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, np.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_multiply(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """
    done
    Args:
        quat1: (4, ) or (n, 4)
        quat2: (4, ) or (n, 4)

    Returns: Hamilton product of two quaternions

    """
    w1, x1, y1, z1 = torch.split(quat1, 1, dim=-1)
    w2, x2, y2, z2 = torch.split(quat2, 1, dim=-1)
    return torch.cat([
        -x1*x2 - y1*y2 - z1*z2 + w1*w2,
        x1*w2 + y1*z2 - z1*y2 + w1*x2,
        -x1*z2 + y1*w2 + z1*x2 + w1*y2,
        x1*y2 - y1*x2 + z1*w2 + w1*z2
    ], dim=-1)


def quaternion_conjugate(quat):
    """
    Return conjugate of quaternion. done
    """
    q_scalar, q_vec = torch.split(quat, [1, 3], dim=-1)
    return torch.cat((q_scalar, -q_vec), dim=-1)


def quaternion_inverse(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> np.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)


def quaternion_real(quaternion):
    """Return real part of quaternion.

    >>> quaternion_real([3, 0, 1, 2])
    3.0

    """
    return float(quaternion[0])


def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([ 0.,  1.,  2.])

    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)


def quaternion_exp(quat: np.ndarray) -> np.ndarray:
    """

    Args:
        quat: (4, ) or (n, 4)

    Returns: (n, 4)
    """
    if len(quat.shape) == 1:
        quat = quat.reshape(1, 4)
    tolerance = 1e-17
    vec = quat[:, 1:]
    v_norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    idx = np.where(v_norm > tolerance)[0]
    vec[idx] = vec[idx] / v_norm[idx]
    magnitude = np.exp(quat[:, [0]])
    return np.concatenate([magnitude * np.cos(v_norm), magnitude * np.sin(v_norm) * vec], axis=-1)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.

    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0)
    >>> np.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2, math.acos(-np.dot(q0, q1)) / angle)
    True

    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quaternion_rot_diff(quat0, quat1):
    """
    q1 = q_diff * q_0
    """
    return quaternion_multiply(quat1, quaternion_conjugate(quat0))


# def quaternion_angular_diff(quat0, quat1):
#     inner_product = np.clip(np.dot(quat0, quat1), -1, 1)
#     theta = 2 * np.arccos(inner_product)
#     return theta


def quaternion_average(quat: torch.Tensor, weights: torch.Tensor = None, keepdim: bool = False):
    """
    compute the weighted average quaternion done

    Args:
        quat: (n_quats, 4)
        weights: weight coefficients
        keepdim:

    Returns: (4, )

    """
    if len(quat.shape) < 2:
        raise ValueError("quat should have shape (batch, 4, ...)")
    if weights is None:
        result = torch.linalg.eigh(torch.einsum('ij...,ik...->...jk', quat, quat))[1][:, -1]
    else:
        if quat.shape[0] != len(weights):
            raise ValueError("number of quaternions and weights mismatch!")
        result = torch.linalg.eigh(torch.einsum('ij...,ik...,i->...jk', quat, quat, weights))[1][:, -1]
    if len(result.size()) == 2:
        result = result.transpose(1, 0)
    if keepdim:
        return result.unsqueeze(0)
    else:
        return result


def quaternion_rotate(ori_quat, rot_quat):
    return quaternion_multiply(quaternion_multiply(rot_quat, ori_quat), quaternion_conjugate(rot_quat))


def quaternion_normalize(quat: torch.Tensor):
    """
    normalize the quaternion done

    Args:
        quat: (4, ) or (n, 4)

    Returns: normalized quaternion

    """
    if len(quat.shape) == 1:
        quat = quat.unsqueeze(0)

    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    idx = torch.where(norm > _EPS)[0]
    quat[idx] = quat[idx] / norm[idx]
    return quat.squeeze()


def random_quaternion(rand=None):
    """Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = random_quaternion()
    >>> np.allclose(1, vector_norm(q))
    True
    >>> q = random_quaternion(np.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)

    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2)*r2, np.sin(t1)*r1,
                        np.cos(t1)*r1, np.sin(t2)*r2])


def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.

    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix()
    >>> np.allclose(np.dot(R.T, R), np.identity(4))
    True

    """
    return quaternion_matrix(random_quaternion(rand))


class Arcball(object):
    """Virtual Trackball Control.

    >>> ball = Arcball()
    >>> ball = Arcball(initial=np.identity(4))
    >>> ball.place([320, 320], 320)
    >>> ball.down([500, 250])
    >>> ball.drag([475, 275])
    >>> R = ball.matrix()
    >>> np.allclose(np.sum(R), 3.90583455)
    True
    >>> ball = Arcball(initial=[1, 0, 0, 0])
    >>> ball.place([320, 320], 320)
    >>> ball.setaxes([1, 1, 0], [-1, 1, 0])
    >>> ball.constrain = True
    >>> ball.down([400, 200])
    >>> ball.drag([200, 400])
    >>> R = ball.matrix()
    >>> np.allclose(np.sum(R), 0.2055924)
    True
    >>> ball.next()

    """
    def __init__(self, initial=None):
        """Initialize virtual trackball control.

        initial : quaternion or rotation matrix

        """
        self._axis = None
        self._axes = None
        self._radius = 1.0
        self._center = [0.0, 0.0]
        self._vdown = np.array([0.0, 0.0, 1.0])
        self._constrain = False
        if initial is None:
            self._qdown = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            initial = np.array(initial, dtype=np.float64)
            if initial.shape == (4, 4):
                self._qdown = quaternion_from_matrix(initial)
            elif initial.shape == (4, ):
                initial /= vector_norm(initial)
                self._qdown = initial
            else:
                raise ValueError("initial not a quaternion or matrix")
        self._qnow = self._qpre = self._qdown

    def place(self, center, radius):
        """Place Arcball, e.g. when window size changes.

        center : sequence[2]
            Window coordinates of trackball center.
        radius : float
            Radius of trackball in window coordinates.

        """
        self._radius = float(radius)
        self._center[0] = center[0]
        self._center[1] = center[1]

    def setaxes(self, *axes):
        """Set axes to constrain rotations."""
        if axes is None:
            self._axes = None
        else:
            self._axes = [unit_vector(axis) for axis in axes]

    @property
    def constrain(self):
        """Return state of constrain to axis mode."""
        return self._constrain

    @constrain.setter
    def constrain(self, value):
        """Set state of constrain to axis mode."""
        self._constrain = bool(value)

    def down(self, point):
        """Set initial cursor window coordinates and pick constrain-axis."""
        self._vdown = arcball_map_to_sphere(point, self._center, self._radius)
        self._qdown = self._qpre = self._qnow
        if self._constrain and self._axes is not None:
            self._axis = arcball_nearest_axis(self._vdown, self._axes)
            self._vdown = arcball_constrain_to_axis(self._vdown, self._axis)
        else:
            self._axis = None

    def drag(self, point):
        """Update current cursor window coordinates."""
        vnow = arcball_map_to_sphere(point, self._center, self._radius)
        if self._axis is not None:
            vnow = arcball_constrain_to_axis(vnow, self._axis)
        self._qpre = self._qnow
        t = np.cross(self._vdown, vnow)
        if np.dot(t, t) < _EPS:
            self._qnow = self._qdown
        else:
            q = [np.dot(self._vdown, vnow), t[0], t[1], t[2]]
            self._qnow = quaternion_multiply(q, self._qdown)

    def next(self, acceleration=0.0):
        """Continue rotation in direction of last drag."""
        q = quaternion_slerp(self._qpre, self._qnow, 2.0+acceleration, False)
        self._qpre, self._qnow = self._qnow, q

    def matrix(self):
        """Return homogeneous rotation matrix."""
        return quaternion_matrix(self._qnow)


def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0*v0 + v1*v1
    if n > 1.0:
        # position outside of sphere
        n = math.sqrt(n)
        return np.array([v0/n, v1/n, 0.0])
    else:
        return np.array([v0, v1, math.sqrt(1.0 - n)])


def arcball_constrain_to_axis(point, axis):
    """Return sphere point perpendicular to axis."""
    v = np.array(point, dtype=np.float64, copy=True)
    a = np.array(axis, dtype=np.float64, copy=True)
    v -= a * np.dot(a, v)  # on plane
    n = vector_norm(v)
    if n > _EPS:
        if v[2] < 0.0:
            np.negative(v, v)
        v /= n
        return v
    if a[2] == 1.0:
        return np.array([1.0, 0.0, 0.0])
    return unit_vector([-a[1], a[0], 0.0])


def arcball_nearest_axis(point, axes):
    """Return axis, which arc is nearest to point."""
    point = np.array(point, dtype=np.float64, copy=False)
    nearest = None
    mx = -1.0
    for axis in axes:
        t = np.dot(arcball_constrain_to_axis(point, axis), point)
        if t > mx:
            nearest = axis
            mx = t
    return nearest


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = np.random.random(3)
    >>> n = vector_norm(v)
    >>> np.allclose(n, np.linalg.norm(v))
    True
    >>> v = np.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> v = np.random.rand(5, 4, 3)
    >>> n = np.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def random_vector(size):
    """Return array of random doubles in the half-open interval [0.0, 1.0).

    >>> v = random_vector(10000)
    >>> np.all(v >= 0) and np.all(v < 1)
    True
    >>> v0 = random_vector(10)
    >>> v1 = random_vector(10)
    >>> np.any(v0 == v1)
    False

    """
    return np.random.random(size)


def vector_product(v0, v1, axis=0):
    """Return vector perpendicular to vectors.

    >>> v = vector_product([2, 0, 0], [0, 3, 0])
    >>> np.allclose(v, [0, 0, 6])
    True
    >>> v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    >>> v1 = [[3], [0], [0]]
    >>> v = vector_product(v0, v1)
    >>> np.allclose(v, [[0, 0, 0, 0], [0, 0, 6, 6], [0, -6, 0, -6]])
    True
    >>> v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    >>> v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    >>> v = vector_product(v0, v1, axis=1)
    >>> np.allclose(v, [[0, 0, 6], [0, -6, 0], [6, 0, 0], [0, -6, 6]])
    True

    """
    return np.cross(v0, v1, axis=axis)


def angle_between_vectors(v0, v1, directed=True, axis=0):
    """Return angle between vectors.

    If directed is False, the input vectors are interpreted as undirected axes,
    i.e. the maximum angle is pi/2.

    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
    >>> np.allclose(a, math.pi)
    True
    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3], directed=False)
    >>> np.allclose(a, 0)
    True
    >>> v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    >>> v1 = [[3], [0], [0]]
    >>> a = angle_between_vectors(v0, v1)
    >>> np.allclose(a, [0, 1.5708, 1.5708, 0.95532])
    True
    >>> v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    >>> v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    >>> a = angle_between_vectors(v0, v1, axis=1)
    >>> np.allclose(a, [1.5708, 1.5708, 1.5708, 0.95532])
    True

    """
    v0 = np.array(v0, dtype=np.float64, copy=False)
    v1 = np.array(v1, dtype=np.float64, copy=False)
    dot = np.sum(v0 * v1, axis=axis)
    dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot if directed else np.fabs(dot))


def quaternion_angular_diff(q1, q2):
    """
    Computes the angle between two quaternions [w,x,y,z].
    theta = arccos(2 * <q1, q2>^2 - 1)
    See https://math.stackexchange.com/questions/90081/quaternion-distance
    """
    theta = 2 * torch.arccos(torch.clip(2 * torch.einsum("bi...,bi...->b...", q1, q2) ** 2 - 1, -1, 1))
    return theta


def angle_between_matrix(m1, m2):
    q1 = quaternion_from_matrix(m1)
    q2 = quaternion_from_matrix(m2)
    return angle_between_quaternion(q1, q2)


def dist_between_matrix(m1, m2):
    return np.linalg.norm(m1[0:3, 3] - m2[0:3, 3])


def inverse_matrix(matrix):
    """Return inverse of square transformation matrix.

    >>> M0 = random_rotation_matrix()
    >>> M1 = inverse_matrix(M0.T)
    >>> np.allclose(M1, np.linalg.inv(M0.T))
    True
    >>> for size in range(1, 7):
    ...     M0 = np.random.rand(size, size)
    ...     M1 = inverse_matrix(M0)
    ...     if not np.allclose(M1, np.linalg.inv(M0)): print(size)

    """
    return np.linalg.inv(matrix)


def invert_transform(matrix):
    # R|t inverse
    inv = np.identity(4)

    rot = matrix[0:3, 0:3]
    inv[0:3, 0:3] = rot.T
    inv[0:3, 3] = -1.0 * numpy.transpose(rot).dot(matrix[0:3, 3])

    return inv


def concatenate_matrices(*matrices):
    """Return concatenation of series of transformation matrices.

    >>> M = np.random.rand(16).reshape((4, 4)) - 0.5
    >>> np.allclose(M, concatenate_matrices(M))
    True
    >>> np.allclose(np.dot(M, M.T), concatenate_matrices(M, M.T))
    True

    """
    M = np.identity(4)
    for i in matrices:
        M = np.dot(M, i)
    return M


def is_same_transform(matrix0, matrix1):
    """Return True if two matrices perform same transformation.

    >>> is_same_transform(np.identity(4), np.identity(4))
    True
    >>> is_same_transform(np.identity(4), random_rotation_matrix())
    False

    """
    matrix0 = np.array(matrix0, dtype=np.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = np.array(matrix1, dtype=np.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return np.allclose(matrix0, matrix1)


def is_same_quaternion(q0, q1):
    """Return True if two quaternions are equal."""
    q0 = np.array(q0)
    q1 = np.array(q1)
    return np.allclose(q0, q1) or np.allclose(q0, -q1)


def angular_velocities_from_quat(q1, q2, dt):
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])


def angular_velocities_from_dq(quat_derivative: torch.Tensor, quat: torch.Tensor):
    """
    done
    Given the derivative (quat_derivative) of quaternion (quat), compute the angular velocity
    Args:
        quat_derivative: (4, ) or (n, 4)
        quat: (4, ) or (n, 4)

    Returns: (4, ) or (n, 4)

    """
    return 2 * quaternion_multiply(quaternion_conjugate(quat), quat_derivative)


def angular_velocity_traj_from_quat(q: np.ndarray, dt: Union[float, np.ndarray]):
    """
    given a quaternion trajectory, compute the angular velocity trajectory

    Args:
        q: (T, 4)
        dt: (T, ) or float. If a float value is given, it will be repeated

    Returns: (T, 3) angular velocity trajectory

    """
    if isinstance(dt, float):
        dt = np.ones(q.shape[0]-1) * dt
    vel_traj = []
    for t in range(q.shape[0]-1):
        vel_traj.append(angular_velocities_from_quat(q[t], q[t+1], dt[t]))
    vel_traj.append(np.zeros(3))
    return np.array(vel_traj)


def integrate_angular_vel_to_quat(w: np.ndarray, q: np.ndarray, dt: float, exp: bool = True) -> np.ndarray:
    """
    TODO implement for batch input
    Args:
        w: (3, ) or (n, 3)
        q: (4, ) or (n, 4)
        dt: delta time
        exp: whether to use exponential map

    Returns: (n, 4) the new quaternion with the integrated angular velocity
    """
    if len(w.shape) == 1:
        w = w.reshape(1, 3)
    n = w.shape[0]
    if len(q.shape) == 1:
        q = q.reshape(1, 4)
    if exp:

        w = np.concatenate((np.zeros((n, 1)), w), axis=-1)
        hat_q = quaternion_exp(0.5 * dt * w)
        return np.array([
            quaternion_multiply(q0, q.flatten()) for q0 in hat_q
        ])
    else:
        skew = np.array([
            np.array([
                [   0, -v[0], -v[1], -v[2]],
                [v[0],     0,  v[2], -v[1]],
                [v[1], -v[2],     0,  v[0]],
                [v[2],  v[1], -v[0],     0]
            ]) for v in w
        ])
        d_quat = 0.5 * np.einsum("bij,bj->bi", skew, q)
        return q + d_quat * dt


def _import_module(name, package=None, warn=True, prefix='_py_', ignore='_'):
    """Try import all public attributes from module into global namespace.

    Existing attributes with name clashes are renamed with prefix.
    Attributes starting with underscore are ignored by default.

    Return True on successful import.

    """
    import warnings
    from importlib import import_module
    try:
        if not package:
            module = import_module(name)
        else:
            module = import_module('.' + name, package=package)
    except ImportError:
        if warn:
            warnings.warn('failed to import module %s' % name)
    else:
        for attr in dir(module):
            if ignore and attr.startswith(ignore):
                continue
            if prefix:
                if attr in globals():
                    globals()[prefix + attr] = globals()[attr]
                elif warn:
                    warnings.warn('no Python implementation of ' + attr)
            globals()[attr] = getattr(module, attr)
        return True


_import_module('_transformations', warn=False)

if __name__ == '__main__':
    import doctest
    import random  # noqa: used in doctests
    try:
        np.set_printoptions(suppress=True, precision=5, legacy='1.13')
    except TypeError:
        np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
