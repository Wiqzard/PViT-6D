import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import uniform_, xavier_uniform_, constant_

from utils.bbox import xywh_to_xyxy, xyxy_to_xywh
from utils.pose_ops import (
    batch_euler_angles_to_rotation_matrix,
)


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init_(module):
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def inverse_tanh(x, eps=1e-5):
    x = x.clamp(min=-1 + eps, max=1 - eps)
    return 0.5 * torch.log((1 + x) / (1 - x))


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-scale deformable attention.
    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# def get_cdn_group(batch,
#                  num_classes,
#                  num_queries,
#                  class_embed,
#                  num_dn=100,
#                  cls_noise_ratio=0.5,
#                  box_noise_scale=1.0,
#                  training=False):
#    """
#    Get contrastive denoising training group. This function creates a contrastive denoising training group with
#    positive and negative samples from the ground truths (gt). It applies noise to the class labels and bounding
#    box coordinates, and returns the modified labels, bounding boxes, attention mask and meta information.
#
#    Args:
#        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
#            (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
#            indicating the number of gts of each image.
#        num_classes (int): Number of classes.
#        num_queries (int): Number of queries.
#        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
#        num_dn (int, optional): Number of denoising. Defaults to 100.
#        cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
#        box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
#        training (bool, optional): If it's in training mode. Defaults to False.
#
#    Returns:
#        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
#            bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
#            is less than or equal to 0, the function returns None for all elements in the tuple.
#    """
#
#    if (not training) or num_dn <= 0:
#        return None, None, None, None
#
#    batch_idx = batch["batch_idx"]
#    bs = batch["img"].shape[0]
#    gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
#    #gt_groups = batch['gt_groups']
#    total_num = sum(gt_groups)
#    max_nums = max(gt_groups)
#    if max_nums == 0:
#        return None, None, None, None
#
#    num_group = num_dn // max_nums
#    num_group = 1 if num_group == 0 else num_group
#    # pad gt to max_num of a batch
#    bs = len(gt_groups)
#    gt_cls = batch['cls'].long()  # (bs*num, )
#    gt_bbox = batch['bboxes']  # bs*num, 4
#    b_idx = batch['batch_idx'].long()
#
#    # each group has positive and negative queries.
#    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
#    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
#    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )
#
#    # positive and negative mask
#    # (bs*num*num_group, ), the second total_num*num_group part as negative samples
#    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num
#
#    if cls_noise_ratio > 0:
#        # half of bbox prob
#        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
#        idx = torch.nonzero(mask).squeeze(-1)
#        # randomly put a new one here
#        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
#        dn_cls[idx] = new_label
#
#    if box_noise_scale > 0:
#        known_bbox = xywh_to_xyxy(dn_bbox)
#
#        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4
#
#        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
#        rand_part = torch.rand_like(dn_bbox)
#        rand_part[neg_idx] += 1.0
#        rand_part *= rand_sign
#        known_bbox += rand_part * diff
#        known_bbox.clip_(min=0.0, max=1.0)
#        dn_bbox = xyxy_to_xywh(known_bbox)
#        dn_bbox = inverse_sigmoid(dn_bbox)
#
#    # total denoising queries
#    num_dn = int(max_nums * 2 * num_group)
#    # class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])
#    dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
#    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
#    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)
#
#    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
#    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)
#
#    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
#    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
#    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox
#
#    tgt_size = num_dn + num_queries
#    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
#    # match query cannot see the reconstruct
#    attn_mask[num_dn:, :num_dn] = True
#    # reconstruct cannot see each other
#    for i in range(num_group):
#        if i == 0:
#            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
#        if i == num_group - 1:
#            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * i * 2] = True
#        else:
#            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), max_nums * 2 * (i + 1):num_dn] = True
#            attn_mask[max_nums * 2 * i:max_nums * 2 * (i + 1), :max_nums * 2 * i] = True
#    dn_meta = {
#        'dn_pos_idx': [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
#        'dn_num_group': num_group,
#        'dn_num_split': [num_dn, num_queries]}
#
#    return padding_cls.to(class_embed.device), padding_bbox.to(class_embed.device), attn_mask.to(
#        class_embed.device), dn_meta


def get_cdn_group(
    batch,
    num_classes,
    num_queries,
    class_embed,
    num_dn=100,
    cls_noise_ratio=0.5,
    box_noise_scale=1.0,
    z_noise_scale=0.2,
    rot_noise_scale=0.1,
    training=False,
    rot_constructor=None,
):
    """
    Get contrastive denoising training group. This function creates a contrastive denoising training group with
    positive and negative samples from the ground truths (gt). It applies noise to the class labels and bounding
    box coordinates, and returns the modified labels, bounding boxes, attention mask and meta information.

    Args:
        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
            (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
            indicating the number of gts of each image.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
        num_dn (int, optional): Number of denoising. Defaults to 100.
        cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
        training (bool, optional): If it's in training mode. Defaults to False.

    Returns:
        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
            bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
            is less than or equal to 0, the function returns None for all elements in the tuple.
    """

    if (not training) or num_dn <= 0:
        return None, None, None, None, None, None, None
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None, None, None, None
    gt_groups = batch["gt_groups"]

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(gt_groups)
    gt_cls = batch["cls"]  # (bs*num, )
    gt_bbox = batch["bboxes"]  # bs*num, 4
    gt_scale = gt_bbox[..., 2:].max(dim=-1)[0]
    gt_z = batch["poses"][:, -1, -1].unsqueeze(-1)
    gt_rot = batch["poses"][:, :3, :3]  # .reshape(-1, 6)
    b_idx = batch["batch_idx"]

    # each group has positive and negative queries.
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_scale = gt_scale.repeat(2 * num_group, 1)
    dn_z = gt_z.repeat(2 * num_group, 1)
    dn_rot = gt_rot.repeat(2 * num_group, 1, 1)
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # positive and negative mask
    # (bs*num*num_group, ), the second total_num*num_group part as negative samples
    neg_idx = (
        torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device)
        + num_group * total_num
    )

    if cls_noise_ratio > 0:
        # half of bbox prob
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint_like(
            idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device
        )
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        known_bbox = xywh_to_xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(
            1, 2
        ) * box_noise_scale  # 2*num_group*bs*num, 4
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy_to_xywh(known_bbox)
        dn_centroid = inverse_sigmoid(dn_bbox)[..., :2]
        dn_scale = dn_bbox[..., 2:].max(dim=-1)[0].unsqueeze(-1)

    if z_noise_scale > 0:
        rand_sign = torch.randint_like(dn_z, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_z) * z_noise_scale
        rand_part[neg_idx] += 0.2 * dn_z[neg_idx]
        rand_part *= rand_sign
        dn_z += rand_part
        dn_z.clamp_(min=0.0, max=5.0)

    if rot_noise_scale > 0:
        # rand_sign = torch.randint_like(dn_rot, 0, 2) * 2.0 - 1.0
        if rot_constructor is None:
            rand_euler_angles = torch.rand(dn_rot.shape[:-1], device=dn_rot.device)
            rand_euler_angles = (
                rand_euler_angles
                / rand_euler_angles.sum(dim=-1, keepdim=True)
                * rot_noise_scale
                * 2
                * torch.pi
            )
            
            rand_euler_angles[neg_idx] += 0.4 * torch.pi 
            rand_part = batch_euler_angles_to_rotation_matrix(rand_euler_angles).to(dn_rot.device)
            dn_rot = torch.bmm(dn_rot, rand_part)[:, :3, :2].reshape(-1, 6)
        else:
            dn_rot = rot_constructor(dn_rot[:, :3, :3].reshape(-1, 9))
            rand_sign = torch.randint_like(dn_rot, 0, 2) * 2.0 - 1.0
            rand_part = torch.rand_like(dn_rot) * rot_noise_scale
            rand_part[neg_idx] += 0.4 * dn_rot[neg_idx]
            rand_part *= rand_sign
            dn_rot += rand_part


    # total denoising queries
    num_dn = int(max_nums * 2 * num_group)
    # class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])
    dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_centroid = torch.zeros(bs, num_dn, 2, device=gt_bbox.device)
    padding_scale = torch.zeros(bs, num_dn, 1, device=gt_scale.device)
    padding_z = torch.zeros(bs, num_dn, 1, device=gt_z.device)
    padding_rot = torch.zeros(bs, num_dn, dn_rot.shape[-1], device=gt_rot.device)

    map_indices = torch.cat(
        [torch.tensor(range(num), dtype=torch.long) for num in gt_groups]
    )
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_centroid[(dn_b_idx, map_indices)] = dn_centroid
    padding_scale[(dn_b_idx, map_indices)] = dn_scale
    padding_z[(dn_b_idx, map_indices)] = dn_z
    padding_rot[(dn_b_idx, map_indices)] = dn_rot

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_nums * 2 * i : max_nums * 2 * (i + 1),
                max_nums * 2 * (i + 1) : num_dn,
            ] = True
        if i == num_group - 1:
            attn_mask[
                max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2
            ] = True
        else:
            attn_mask[
                max_nums * 2 * i : max_nums * 2 * (i + 1),
                max_nums * 2 * (i + 1) : num_dn,
            ] = True
            attn_mask[
                max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i
            ] = True
    dn_meta = {
        "dn_pos_idx": [
            p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)
        ],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_centroid.to(class_embed.device),
        padding_scale.to(class_embed.device),
        padding_z,
        padding_rot,
        attn_mask.to(class_embed.device),
        dn_meta,
    )
