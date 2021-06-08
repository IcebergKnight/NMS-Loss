import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES
import numpy as np

def gen_group_inds(gt_inds):
    target = gt_inds.cpu().numpy()
    gt_label = np.unique(target)
    res = []
    for i in gt_label:
        res.append((np.argwhere(target == i).reshape(-1)))
    return res



@weighted_loss
def tag_wing_loss(pred, extras):
    (gt_inds, anchor_inds, group_anchor, alpha, beta) = extras
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    loss = 0
    for (img_pred, img_gt_inds, img_anchor_inds) in zip(pred, gt_inds, anchor_inds):
        if group_anchor:
            anchor_group = gen_group_inds(img_anchor_inds)
            single_img_loss = 0
            for same_anchor_inds in anchor_group:
                single_img_loss += single_tag_loss(img_pred[same_anchor_inds], 
                                        img_gt_inds[same_anchor_inds], alpha, beta) 
            single_img_loss /= len(anchor_group)
        else:
            single_img_loss = single_tag_loss(img_pred, img_gt_inds, alpha, beta)
        loss += single_img_loss
    loss /= img_num
    return loss

def wing_loss(diff, alpha, beta):
    C = alpha - alpha * np.log(1 + alpha / beta)
    diff_abs = diff.abs().reshape(-1)
    loss = diff_abs.clone()
    idx_smaller = diff_abs < alpha
    idx_bigger = diff_abs >= alpha
    loss[idx_smaller] = alpha * torch.log(1 + diff_abs[idx_smaller] / beta)
    loss[idx_bigger]  = loss[idx_bigger] - C
    loss = loss.mean()
    return loss

def single_tag_loss(pred, gt_inds, alpha, beta):
    assert len(pred) == len(gt_inds)
    assert pred.numel() >= 2
    inds = gen_group_inds(gt_inds)
    # used for there are only negative samples
    if len(inds) == 1:
        return torch.mean(pred) * 0

    eps = 1e-6
    tags = []
    pull = 0

    # discard negative samples:
    sel_inds = []
    for ind in inds:
        if gt_inds[ind[0]] == -1: # when it is negative
            continue
        sel_inds.append(ind)
    inds = sel_inds

    for ind in inds:
        group = pred[ind]
        tags.append(torch.mean(group, dim=0))
        pull += wing_loss((group - tags[-1].expand_as(group)), alpha, beta)

    tags = torch.stack(tags)
    num = tags.size()[0]
    size = (num, num, tags.size()[1])
    A = tags.unsqueeze(dim=1).expand(*size)
    B = A.permute(1, 0, 2)
    diff = A - B
    diff = torch.pow(diff, 2).sum(dim=2)
    # print(diff)
    push = torch.exp(-diff)
    push = torch.sum(push) - num

    push = push/((num - 1) * num + eps) * 0.5
    pull = pull/(num + eps)
    loss = push + pull
    return loss


@LOSSES.register_module
class TagWingLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, group_anchor=True, 
                 alpha = 10, beta = 2):
        super(TagWingLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.group_anchor = group_anchor
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                pred,
                gt_inds,
                anchor_inds,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_tag = self.loss_weight * tag_wing_loss(
            pred,
            (gt_inds, anchor_inds, self.group_anchor, self.alpha, self.beta),
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_tag
