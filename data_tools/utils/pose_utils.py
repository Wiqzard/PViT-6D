import math

import numpy as np


def axangle2mat(axis, angle, is_normalized=False):
    """Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x * x + y * y + z * z)
        x = x / n
        y = y / n
        z = z / n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    return np.array(
        [
            [x * xC + c, xyC - zs, zxC + ys],
            [xyC + zs, y * yC + c, yzC - xs],
            [zxC - ys, yzC + xs, z * zC + c],
        ]
    )


def allocentric_to_egocentric(allo_pose, cam_ray=(0, 0, 1.0)):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.
    Since objects further away from the optical center undergo skewing,
    we try to visually correct by rotating back the amount between
    optical center ray and object centroid ray. Another way to solve
    that might be translational variance
    (https://arxiv.org/abs/1807.03247)
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    trans = allo_pose[:3, 3]
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount

    if angle > 0:
        ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
        ego_pose[:3, 3] = trans
        rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
        ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
    else:
        ego_pose = allo_pose.copy()
    return ego_pose
