from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import math


def get_rot_mat(rot, rot_type):
    if rot_type in ["ego_quat", "allo_quat"]:
        rot_m = quat2mat_torch(rot)
    elif rot_type in ["ego_rot6d", "allo_rot6d"]:
        rot_m = rot6d_to_mat_batch(rot)
    elif rot_type in ["allo_axis_angle", "ego_axis_angle"]:
        rot_m = axis_angle_to_mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m


def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    norm_quat = quat / (norm_quat + eps)
    qw, qx, qy, qz = (
        norm_quat[:, 0],
        norm_quat[:, 1],
        norm_quat[:, 2],
        norm_quat[:, 3],
    )
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [
            1.0 - (yY + zZ),
            xY - wZ,
            xZ + wY,
            xY + wZ,
            1.0 - (xX + zZ),
            yZ - wX,
            xZ - wY,
            yZ + wX,
            1.0 - (xX + yY),
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def rot6d_to_mat_batch(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix.
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks. CVPR 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # poses
    x_raw = d6[..., 0:3]  # bx3
    y_raw = d6[..., 3:6]  # bx3

    x = F.normalize(x_raw, p=2, dim=-1)  # bx3
    z = torch.cross(x, y_raw, dim=-1)  # bx3
    z = F.normalize(z, p=2, dim=-1)  # bx3
    y = torch.cross(z, x, dim=-1)  # bx3

    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)  # (b,3,3)


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    """
    Args:
        translation: Nx3
        rot_allo: Nx3x3
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor(
        [0, 0, 1.0], dtype=translation.dtype, device=translation.device
    )  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego


def pose_from_centroid_z(
    rot,
    centroid,
    z,
    cam,
    im_h: int,
    im_w: int,
    eps=1e-4,
    is_allo=True,
    detach=False,
):
    """
    Calculate pose from centroid and z.
    Args:
        rot: Nx3x3
        centroid: Nx2
        z: Nx1
        cam: Nx3
    """
    if cam.dim() == 2:
        cam = cam.unsqueeze(0).repeat(rot.shape[0], 1, 1)
        cam.unsqueeze_(0)
    if z.ndim == 0:
        z = z[None]

    cx = centroid[:, 0, None] * im_w
    cy = centroid[:, 1, None] * im_h
    trans = torch.cat(
        [
            z[:, None] * (cx - cam[:, 0, 2, None]) / cam[:, 0, 0, None],
            z[:, None] * (cy - cam[:, 1, 2, None]) / cam[:, 1, 1, None],
            z[:, None],
        ],
        dim=-1,
    )

    rot_ego = rot
    if is_allo:
        trans_ = trans.detach() if detach else trans
        rot_ego = allo_to_ego_mat_torch(trans_, rot, eps=eps)
    return rot_ego, trans


def axis_angle_to_mat_batch(axes_angles):
    """
    Args:
        axis_angle: Nx4
    Returns:
        rot_m: Nx3x3
    """
    axes = axes_angles[:, 0:3]
    angles = axes_angles[:, 3:4]
    batch_size = axes.size(0)

    # Ensure the axes are unit vectors
    axes = axes / torch.norm(axes, dim=1, keepdim=True)

    zeros = torch.zeros(batch_size, device=axes.device)
    K_matrices = torch.stack(
        [
            torch.stack([zeros, -axes[:, 2], axes[:, 1]], dim=1),
            torch.stack([axes[:, 2], zeros, -axes[:, 0]], dim=1),
            torch.stack([-axes[:, 1], axes[:, 0], zeros], dim=1),
        ],
        dim=1,
    )

    eye_batch = torch.eye(3, device=axes.device).expand(batch_size, 3, 3)
    sin_angles = torch.sin(angles).view(-1, 1, 1)
    cos_angles = torch.cos(angles).view(-1, 1, 1)
    R_matrices = (
        eye_batch
        + sin_angles * K_matrices
        + (1 - cos_angles) * torch.matmul(K_matrices, K_matrices)
    )
    return R_matrices


def pose_from_pred_centroid_z(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-4,
    is_allo=True,
    z_type="REL",
):
    """
    Args:
        pred_rots:
        pred_centroids:
        pred_z_vals: [B, 1]
        roi_cams: absolute cams
        roi_centers:
        roi_scales:
        roi_whs: (bw,bh) for bboxes
        eps:
        is_allo:
        z_type: REL | ABS | LOG | NEG_LOG
    Returns:
    """
    if cams.dim() == 2:
        cams.unsqueeze_(0)
    assert cams.dim() == 3, cams.dim()
    c = torch.stack(
        [
            (pred_centroids[:, 0] * roi_whs[:, 0]) + roi_centers[:, 0],
            (pred_centroids[:, 1] * roi_whs[:, 1]) + roi_centers[:, 1],
        ],
        dim=1,
    )

    cx = c[:, 0:1]
    cy = c[:, 1:2]

    # unnormalize regressed z
    if z_type == "abs":
        z = pred_z_vals
    elif z_type == "rel":
        # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
        z = pred_z_vals * resize_ratios.view(-1, 1)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """

    translation = torch.cat(
        [
            z * (cx - cams[:, 0:1, 2]) / cams[:, 0:1, 0],
            z * (cy - cams[:, 1:2, 2]) / cams[:, 1:2, 1],
            z,
        ],
        dim=1,
    )

    if pred_rots.ndim == 3 and pred_rots.shape[-1] == 3:  # Nx3x3
        if is_allo:
            rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)
        else:
            rot_ego = pred_rots
    return rot_ego, translation


def pose_from_pred(pred_rots, pred_transes, eps=1e-4, is_allo=True):
    translation = pred_transes
    if is_allo:
        rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)
    else:
        rot_ego = pred_rots
    return rot_ego, translation


def re(R_est, R_gt):
    """Rotational Error.
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    return rd_deg


def get_closest_rot(rot_est, rot_gt, sym_info):
    """get the closest rot_gt given rot_est and sym_info.
    rot_est: ndarray
    rot_gt: ndarray
    sym_info: None or Kx3x3 ndarray, m2m
    """
    if sym_info is None:
        return rot_gt
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    # find the closest rot_gt with smallest re
    r_err = re(rot_est, rot_gt)
    closest_rot_gt = rot_gt
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        rot_gt_sym = rot_gt.dot(sym_info[i])
        cur_re = re(rot_est, rot_gt_sym)
        if cur_re < r_err:
            r_err = cur_re
            closest_rot_gt = rot_gt_sym

    return closest_rot_gt


def get_closest_rot_batch(pred_rots, gt_rots, sym_infos):
    """
    get closest gt_rots according to current predicted poses_est and sym_infos
    --------------------
    pred_rots: [B, 4] or [B, 3, 3]
    gt_rots: [B, 4] or [B, 3, 3]
    sym_infos: list [Kx3x3 or None],
        stores K rotations regarding symmetries, if not symmetric, None
    -----
    closest_gt_rots: [B, 3, 3]
    """
    batch_size = pred_rots.shape[0]
    device = pred_rots.device
    if pred_rots.shape[-1] == 4:
        pred_rots = quat2mat_torch(pred_rots[:, :4])
    if gt_rots.shape[-1] == 4:
        gt_rots = quat2mat_torch(gt_rots[:, :4])

    closest_gt_rots = gt_rots.clone().cpu().numpy()  # B,3,3

    for i in range(batch_size):
        closest_rot = get_closest_rot(
            pred_rots[i].detach().cpu().numpy(),
            gt_rots[i].cpu().numpy(),
            sym_infos[i],
        )
        # TODO: automatically detect rot_gt's format in PM_Loss to avoid converting multiple times
        closest_gt_rots[i] = closest_rot
    closest_gt_rots = torch.tensor(closest_gt_rots, device=device, dtype=gt_rots.dtype)
    return closest_gt_rots


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
    R += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def batch_rotation_matrix_to_euler_angles(R):
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-6

    x = torch.where(
        ~singular,
        torch.atan2(R[:, 2, 1], R[:, 2, 2]),
        torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    )
    
    y = torch.where(
        ~singular,
        torch.atan2(-R[:, 2, 0], sy),
        torch.atan2(-R[:, 2, 0], sy)
    )
    
    z = torch.where(
        ~singular,
        torch.atan2(R[:, 1, 0], R[:, 0, 0]),
        torch.zeros_like(sy)
    )
    
    return torch.stack((x, y, z), dim=1)


def batch_rotation_matrix_to_quaternion(R):
    qw = torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) / 2
    qx = (R[:, 2, 1] - R[:, 1, 2]) / (4 * qw)
    qy = (R[:, 0, 2] - R[:, 2, 0]) / (4 * qw)
    qz = (R[:, 1, 0] - R[:, 0, 1]) / (4 * qw)
    return torch.stack((qx, qy, qz, qw), dim=1)

def batch_add_noise_to_quaternion(quaternion, noise_std_dev):
    noise = torch.normal(0.0, noise_std_dev, size=quaternion.shape)
    return quaternion + noise

def batch_quaternion_to_rotation_matrix(quaternion):
    norm = torch.sqrt(torch.sum(quaternion ** 2, dim=-1, keepdim=True))
    quaternion = quaternion / norm
    qx, qy, qz, qw = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    R = torch.stack((
        torch.stack((1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw), dim=1),
        torch.stack((2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw), dim=1),
        torch.stack((2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2), dim=1)
    ), dim=1)

    return R

def batch_euler_angles_to_rotation_matrix(angles):
    batch_size = angles.shape[0]
    alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]
    # Precompute sin and cos values
    cos_alpha, sin_alpha = torch.cos(alpha), torch.sin(alpha)
    cos_beta, sin_beta = torch.cos(beta), torch.sin(beta)
    cos_gamma, sin_gamma = torch.cos(gamma), torch.sin(gamma)
    # Create rotation matrices using the Euler angles (Z-Y-X convention used here)
    R = torch.zeros((batch_size, 3, 3))
    R[:, 0, 0] = cos_alpha*cos_beta
    R[:, 0, 1] = cos_alpha*sin_beta*sin_gamma - sin_alpha*cos_gamma
    R[:, 0, 2] = cos_alpha*sin_beta*cos_gamma + sin_alpha*sin_gamma
    R[:, 1, 0] = sin_alpha*cos_beta
    R[:, 1, 1] = sin_alpha*sin_beta*sin_gamma + cos_alpha*cos_gamma
    R[:, 1, 2] = sin_alpha*sin_beta*cos_gamma - cos_alpha*sin_gamma
    R[:, 2, 0] = -sin_beta
    R[:, 2, 1] = cos_beta*sin_gamma
    R[:, 2, 2] = cos_beta*cos_gamma
    return R