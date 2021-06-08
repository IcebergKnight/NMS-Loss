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
def tag_point_loss(pred, extras):
    (gt_inds, anchor_inds, group_anchor) = extras
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    loss = 0
    for (img_pred, img_gt_inds, img_anchor_inds) in zip(pred, gt_inds, anchor_inds):
        if group_anchor:
            anchor_group = gen_group_inds(img_anchor_inds)
            single_img_loss = 0
            for same_anchor_inds in anchor_group:
                single_img_loss += single_tag_loss(img_pred[same_anchor_inds], 
                                        img_gt_inds[same_anchor_inds]) 
            single_img_loss /= len(anchor_group)
        else:
            single_img_loss = single_tag_loss(img_pred, img_gt_inds)
        loss += single_img_loss
    loss /= img_num
    return loss


def single_tag_loss(pred, gt_inds):
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
        pull += torch.mean((group - tags[-1].expand_as(group))**2)
    obj_num = len(tags)
    pull /= (obj_num + eps)

    push = 0
    if obj_num > 1:
        for idx, ind in enumerate(inds):
            group = pred[ind]
            other_tags = tags[:idx] + tags[idx + 1:]
            other_tags = torch.stack(other_tags)
            size = (group.size(0), other_tags.size(0), other_tags.size(1))
            A = group.unsqueeze(dim=1).expand(*size)
            B = other_tags.unsqueeze(dim=0).expand(*size)
            diff = torch.mean((A - B)**2, dim = 2)
            diff = torch.clamp(1 - diff, min = 0)
            diff = torch.mean(diff)
            push += diff
        push /= (obj_num + eps)

    loss = push + pull
    return loss


@LOSSES.register_module
class TagPointLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, group_anchor=True):
        super(TagPointLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.group_anchor = group_anchor

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
        loss_tag = self.loss_weight * tag_point_loss(
            pred,
            (gt_inds, anchor_inds, self.group_anchor),
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_tag
