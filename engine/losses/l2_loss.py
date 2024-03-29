import torch
import torch.nn as nn


def l2_loss(pred, target, reduction="mean"):
    assert pred.size() == target.size() and target.numel() > 0
    assert pred.size()[0] == target.size()[0]
    batch_size = pred.size()[0]
    loss = torch.norm((pred - target).view(batch_size, -1), p=2, dim=1, keepdim=True)
    # loss = torch.sqrt(torch.sum(((pred - target)** 2).view(batch_size, -1), 1))
    # print(loss.shape)
    """
    _mse_loss = nn.MSELoss(reduction='none')
    loss_mse = _mse_loss(pred, target)
    print('l2 from mse loss: {}'.format(
        torch.sqrt(
            torch.sum(
                loss_mse.view(batch_size, -1),
                1
            )
        ).mean()))
    """
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class L2Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(L2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = self.loss_weight * l2_loss(pred, target, reduction=self.reduction)
        return loss
