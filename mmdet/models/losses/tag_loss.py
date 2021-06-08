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
def tag_loss(pred, extras):
    (gt_inds, anchor_inds, group_anchor, pull_weight, push_weight) = extras
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    loss = 0
    for (img_pred, img_gt_inds, img_anchor_inds) in zip(pred, gt_inds, anchor_inds):
        if group_anchor:
            anchor_group = gen_group_inds(img_anchor_inds)
            single_img_loss = 0
            for same_anchor_inds in anchor_group:
                single_img_loss += single_tag_loss(img_pred[same_anchor_inds], 
                                        img_gt_inds[same_anchor_inds],
                                        pull_weight, push_weight) 
            single_img_loss /= len(anchor_group)
        else:
            single_img_loss = single_tag_loss(img_pred, img_gt_inds, pull_weight, push_weight)
        loss += single_img_loss
    loss /= img_num
    return loss


def single_tag_loss(pred, gt_inds, pull_weight, push_weight):
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
        pull = pull +  torch.mean((group - tags[-1].expand_as(group))**2)

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
    pull = pull/(num + eps)
    push = push/((num - 1) * num + eps) * 0.5
    loss = push * push_weight + pull * pull_weight
    return loss


@LOSSES.register_module
class TagLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, group_anchor=True, 
                pull_weight = 1, push_weight = 1):
        super(TagLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.group_anchor = group_anchor
        self.pull_weight = pull_weight
        self.push_weight = push_weight

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
        loss_tag = self.loss_weight * tag_loss(
            pred,
            (gt_inds, anchor_inds, self.group_anchor, self.pull_weight, self.push_weight),
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_tag
