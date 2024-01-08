import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_gain=None,
        use_fl=True,
        alpha=0.25,
        gamma=2.0,
    ):
        super().__init__()
        if cost_gain is None:
            cost_gain = {
                "cls": 1,
                "centroid": 3,
                "code": 3,
            }
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_kpts,
        pred_scores,
        pred_codes,
        gt_kpts,
        gt_cls,
        gt_codes,
    ):
        """
        Returns:
            (List[Tuple[Tensor, Tensor]]): A list of size batch_size, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, nq, nc = pred_scores.shape

        # [batch_size * num_queries, num_classes]
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = (
            F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        )
        pred_codes = pred_codes.detach().view(-1, 8)
        # [batch_size * num_queries, 4]
        pred_kpts = pred_kpts.detach().view(-1, 2)
        gt_kpts = gt_kpts.detach().view(-1, 2)
        gt_cls = gt_cls.detach().view(-1)
        c = gt_codes.shape[-1]
        gt_codes = gt_codes.detach().view(-1, c)

        # Compute the classification cost
        #pred_scores = pred_scores[:, gt_cls.long()]
        if self.use_fl:
            neg_cost_class = (
                (1 - self.alpha)
                * (pred_scores**self.gamma)
                * (-(1 - pred_scores + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha
                * ((1 - pred_scores) ** self.gamma)
                * (-(pred_scores + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # Compute the L1 cost between boxes
        cost_kpts = (
            (pred_kpts.unsqueeze(1) - gt_kpts.unsqueeze(0)).abs().sum(-1)
        )  # (bs*num_queries, num_gt)

        # Compute Hamming distance between codes
        cost_codes = (
            (pred_codes.unsqueeze(1) - gt_codes.unsqueeze(0)).abs().sum(-1)/pred_codes.shape[-1]
        )  # (bs*num_queries, num_gt)

        # Final cost matrix
        C = (
            self.cost_gain["cls"] * cost_class
            + self.cost_gain["centroid"] * cost_kpts
            + self.cost_gain["code"] * cost_codes
        )

        C = C.view(bs, nq, -1).cpu()
        # (idx for queries, idx for gt)
        indices = [linear_sum_assignment(c) for c in C]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

#    def get_index(self, match_indices):
#        batch_idx = torch.arange(
#            len(match_indices), device=match_indices[0][0].device
#        ).unsqueeze(-1)
#        src_idx = torch.stack([src for (src, _) in match_indices], dim=0)
#        dst_idx = torch.stack([dst for (_, dst) in match_indices], dim=0)
#        return (batch_idx, src_idx), (batch_idx, dst_idx)


    def get_index(self, match_indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)]
        )
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx
#
# pred = torch.randn(8, 3)
# gt = torch.randn(8, 3)
# out = testl1(pred, gt)
# out = out.reshape(2, 4, -1)
# indices = [linear_sum_assignment(c) for c in out]
# matched_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# matcher = HungarianMatcher()
# pred_kpts = torch.randn(8, 256, 2)
# pred_scores = torch.randn(8, 256, 2)
# pred_code = torch.randn(8, 256, 8)
# gt_kpts = torch.randn(8, 256, 2)
# gt_cls = torch.randint(0, 2, (8, 256))
# gt_codes = torch.randn(8, 256, 8)
# match_indices = matcher(pred_kpts, pred_scores, pred_code, gt_kpts, gt_cls, gt_codes)
# src_idx = torch.stack([src for (src, _) in match_indices], dim=0)
# batch_idx = torch.arange(src_idx.shape[0], device=src_idx.device)
# idx, gt_idx = get_index(match_indices)
#
# rpred_kpts = pred_kpts[idx]
#
# print("halt")
#
