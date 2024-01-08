import torch
import torch.nn as nn
import torch.nn.functional as F

# from pytorch3d.ops import box3d_overlap


def corner_to_aabb(boxes):
    # Convert corner representation to axis-aligned (min, max) representation
    # Input shape: (BS, 8, 3)
    # Output shape: (BS, 6) -> x_min, y_min, z_min, x_max, y_max, z_max
    min_coords = torch.min(boxes, dim=1)[0]
    max_coords = torch.max(boxes, dim=1)[0]
    aabbs = torch.cat([min_coords, max_coords], dim=1)
    return aabbs


def get_3d_iou(boxes1, boxes2):
    """
    Calculate the 3D Intersection-over-Union (IoU) between two sets of boxes.

    Args:
    boxes1, boxes2 (torch.Tensor): Tensors containing box coordinates, expected shape [N, 6]

    Returns:
    torch.Tensor: IoU for each pair of boxes, shape [N]
    """
    # Convert corner coordinates to axis-aligned bounding boxes (AABB)
    boxes1 = corner_to_aabb(boxes1)
    boxes2 = corner_to_aabb(boxes2)
    # Check if both sets of boxes have the same dimensions
    assert (
        boxes1.size() == boxes2.size()
    ), "Dimension mismatch between the two sets of boxes"
    # Calculate overlap in each dimension
    x_overlap = torch.clamp_min(
        torch.min(boxes1[:, 3], boxes2[:, 3]) - torch.max(boxes1[:, 0], boxes2[:, 0]),
        min=0,
    )
    y_overlap = torch.clamp_min(
        torch.min(boxes1[:, 4], boxes2[:, 4]) - torch.max(boxes1[:, 1], boxes2[:, 1]),
        min=0,
    )
    z_overlap = torch.clamp_min(
        torch.min(boxes1[:, 5], boxes2[:, 5]) - torch.max(boxes1[:, 2], boxes2[:, 2]),
        min=0,
    )
    # Calculate the volume of the intersection region
    intersection_volume = x_overlap * y_overlap * z_overlap
    # Calculate the volume of each set of boxes
    vol1 = (
        (boxes1[:, 3] - boxes1[:, 0])
        * (boxes1[:, 4] - boxes1[:, 1])
        * (boxes1[:, 5] - boxes1[:, 2])
    )
    vol2 = (
        (boxes2[:, 3] - boxes2[:, 0])
        * (boxes2[:, 4] - boxes2[:, 1])
        * (boxes2[:, 5] - boxes2[:, 2])
    )
    # Calculate the volume of the union region
    union_volume = vol1 + vol2 - intersection_volume
    # Calculate the IoU
    iou = intersection_volume / torch.clamp_min(union_volume, min=1e-8)
    return iou


def iou3d_loss(
    pred_t,
    pred_r,
    target_t,
    target_r,
    bbox3d,
    trans_only=False,
):
    # transform bbox3d to trans rot
    if not trans_only:
        target_bbox3d = torch.bmm(target_r, bbox3d.transpose(1, 2)).transpose(1, 2)
        pred_bbox3d = torch.bmm(pred_r, bbox3d.transpose(1, 2)).transpose(1, 2)

    target_bbox3d = bbox3d + target_t.unsqueeze(1).repeat(1, 8, 1)
    pred_bbox3d = bbox3d + pred_t.unsqueeze(1).repeat(1, 8, 1)
    ious3d = get_3d_iou(pred_bbox3d.detach(), target_bbox3d)
    return ious3d


def get_loss_class(pred_scores, targets, gt_scores):
    
    # logits: [b, num_classes], gt_class: list[[n, 1]]
    bs, nc = pred_scores.shape[:2]
    one_hot = torch.zeros(
        (bs, nc + 1), dtype=torch.int64, device=targets.device
    )
    one_hot.scatter_(1, targets.unsqueeze(-1), 1)
    one_hot = one_hot[..., :-1]
    gt_scores = gt_scores.view(bs, 1) * one_hot
    ##if fl:
    ##    if num_gts and self.vfl:
    ##        loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
    ##    loss_cls /= max(num_gts, 1) / nq
    loss_cls = (
        nn.BCEWithLogitsLoss(reduction="mean")(pred_scores, gt_scores)#.mean(1).sum()
    )
    return loss_cls


if __name__ == "__main__":
    # test iou3d_loss
    pred_t = torch.randn(2, 3)
    pred_r = torch.randn(2, 3, 3)
    target_t = torch.randn(2, 3)
    target_r = torch.randn(2, 3, 3)
    bbox3d = torch.randn(2, 8, 3)
    loss = iou3d_loss(pred_t, pred_r, target_t, target_r, bbox3d)
    print(loss)


    # test get_loss_class
    pred_scores = torch.randn(2, 21)
    targets = torch.randint(0, 21, (2,))
    gt_scores = torch.randn(2, 1)
    loss = get_loss_class(pred_scores, targets, loss)#gt_scores)
    print(loss)
