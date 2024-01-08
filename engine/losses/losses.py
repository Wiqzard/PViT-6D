from typing import Optional , Tuple , Dict
import torch
import torch.nn as nn

from .l2_loss import L2Loss
#from .mask_losses import weighted_ex_loss_probs, soft_dice_loss
from .pm_loss import PyPMLoss
from .rot_loss import angular_distance, rot_l2_loss

import torch
from torch import nn
from torch import Tensor



def get_losses_names(args) -> list[str]:
    losses_name = []
    if args.pm_lw > 0:
        losses_name.append("pm")
    if args.rot_lw > 0:
        losses_name.append("rot")
    if args.centroid_lw > 0:
        losses_name.append("centroid")
    if args.z_lw > 0:
        losses_name.append("z")
    if args.trans_lw > 0:
        losses_name += ["trans_xy", "trans_z"]
    if args.cls_lw > 0:
        losses_name.append("cls")
    return losses_name



#def compute_mask_visib_loss(
#    args: ConfigType,
#    gt_masks: Dict[str, Tensor],
#    out_mask_vis: Tensor,
#) -> LossDictType:
#    mask_loss_type = args.MASK_LOSS_TYPE
#    gt_mask = gt_masks[args.MASK_LOSS_GT]
#    if mask_loss_type == "L1":
#        loss_mask = nn.L1Loss(reduction="mean")(out_mask_vis[:, 0, :, :], gt_mask)
#    elif mask_loss_type == "BCE":
#        loss_mask = nn.BCEWithLogitsLoss(reduction="mean")(
#            out_mask_vis[:, 0, :, :], gt_mask
#        )
#    elif mask_loss_type == "RW_BCE":
#        loss_mask = weighted_ex_loss_probs(
#            torch.sigmoid(out_mask_vis[:, 0, :, :]), gt_mask, weight=None
#        )
#    elif mask_loss_type == "dice":
#        loss_mask = soft_dice_loss(
#            torch.sigmoid(out_mask_vis[:, 0, :, :]),
#            gt_mask,
#            eps=0.002,
#            reduction="mean",
#        )
#    elif mask_loss_type == "CE":
#        loss_mask = nn.CrossEntropyLoss(reduction="mean")(out_mask_vis, gt_mask.long())
#    else:
#        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
#    loss_mask *= args.MASK_LW
#    return loss_mask
#

#def compute_mask_full_loss(
#    args: ConfigType,
#    gt_mask_full: Optional[Tensor],
#    out_mask_full: Optional[Tensor],
#) -> LossDictType:
#    full_mask_loss_type = args.FULL_MASK_LOSS_TYPE
#    if full_mask_loss_type == "L1":
#        loss_mask_full = nn.L1Loss(reduction="mean")(
#            out_mask_full[:, 0, :, :], gt_mask_full
#        )
#    elif full_mask_loss_type == "BCE":
#        loss_mask_full = nn.BCEWithLogitsLoss(reduction="mean")(
#            out_mask_full[:, 0, :, :], gt_mask_full
#        )
#    elif full_mask_loss_type == "RW_BCE":
#        loss_mask_full = weighted_ex_loss_probs(
#            torch.sigmoid(out_mask_full[:, 0, :, :]), gt_mask_full, weight=None
#        )
#    elif full_mask_loss_type == "dice":
#        loss_mask_full = soft_dice_loss(
#            torch.sigmoid(out_mask_full[:, 0, :, :]),
#            gt_mask_full,
#            eps=0.002,
#            reduction="mean",
#        )
#    elif full_mask_loss_type == "CE":
#        loss_mask_full = nn.CrossEntropyLoss(reduction="mean")(
#            out_mask_full, gt_mask_full.long()
#        )
#    else:
#        raise NotImplementedError(f"unknown mask loss type: {full_mask_loss_type}")
#    loss_mask_full *= args.FULL_MASK_LW
#    return loss_mask_full



def compute_point_matching_loss(
    args,
    out_rot: Optional[Tensor],
    gt_rot: Optional[Tensor],
    gt_points: Optional[Tensor],
    out_trans: Optional[Tensor],
    gt_trans: Optional[Tensor],
    diameter: Optional[Tensor],
    sym_infos: Optional[Tensor],
):
    loss_func = PyPMLoss(
        loss_type="l1",
        beta=1.0,
        reduction="mean",
        loss_weight=args.pm_lw,
        norm_by_diameter=args.pm_norm_by_diameter,
        symmetric=args.pm_loss_sym,
        disentangle_t=args.pm_disentangle_t,
        disentangle_z=args.pm_disentangle_z,
        t_loss_use_points=args.pm_t_use_points,
        r_only=args.pm_r_only,
    )
    loss_pm_dict = loss_func(
        pred_rots=out_rot,
        gt_rots=gt_rot,
        points=gt_points,
        pred_transes=out_trans,
        gt_transes=gt_trans,
        diameter=diameter,
        sym_infos=sym_infos,
    )
    return loss_pm_dict


def compute_rot_loss(
    args, out_rot: Optional[Tensor], gt_rot: Optional[Tensor]
):
    if args.rot_loss_type == "angular":
        loss_rot = angular_distance(out_rot, gt_rot)
    elif args.rot_loss_type == "l2":
        loss_rot = rot_l2_loss(out_rot, gt_rot)
    else:
        raise ValueError(f"Unknown rot loss type: {args.rot_loss_type}")
    loss_rot *= args.rot_lw
    return loss_rot


def compute_centroid_loss(
    args, out_centroid: Tensor, gt_trans_ratio: Tensor
):
    if args.centroid_loss_type == "l1":
        loss_centroid = nn.L1Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
    elif args.centroid_loss_type == "l2":
        loss_centroid = L2Loss(reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
    elif args.centroid_loss_type == "mse":
        loss_centroid = nn.MSELoss(reduction="mean")(
            out_centroid, gt_trans_ratio[:, :2]
        )
    else:
        raise ValueError(f"Unknown centroid loss type: {args.centroid_loss_type}")
    loss_centroid *= args.centroid_lw
    return loss_centroid


def compute_z_loss(
    args,
    out_trans_z: Optional[Tensor],
    gt_z: Optional[Tensor],
):
    z_loss_type = args.z_loss_type
    if z_loss_type == "l1":
        loss_z = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
    elif z_loss_type == "l2":
        loss_z = L2Loss(reduction="mean")(out_trans_z, gt_z)
    elif z_loss_type == "mse":
        loss_z = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
    else:
        raise ValueError(f"Unknown z loss type: {z_loss_type}")
    loss_z *= args.z_lw
    return loss_z


def compute_trans_loss(
    args, out_trans: Optional[Tensor], gt_trans: Optional[Tensor]
): 
    if args.TRANS_LOSS_DISENTANGLE:
        if args.TRANS_LOSS_TYPE == "L1":
            loss_trans_xy = nn.L1Loss(reduction="mean")(
                out_trans[:, :2], gt_trans[:, :2]
            )
            loss_trans_z = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
        elif args.TRANS_LOSS_TYPE == "L2":
            loss_trans_xy = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
            loss_trans_z = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
        elif args.TRANS_LOSS_TYPE == "MSE":
            loss_trans_xy = nn.MSELoss(reduction="mean")(
                out_trans[:, :2], gt_trans[:, :2]
            )
            loss_trans_z = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
        else:
            raise ValueError(f"Unknown trans loss type: {args.TRANS_LOSS_TYPE}")
        loss_trans_xy *= args.TRANS_LW
        loss_trans_z *= args.TRANS_LW
        return loss_trans_xy, loss_trans_z
