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
def tag_cosine_loss(pred, extras):
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
                                        pull_weight = pull_weight, 
                                        push_weight = push_weight) 
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
    tag_dim = pred.size(1)
    cos = nn.CosineSimilarity(dim=1, eps=eps)
    pull = 0
    tags = []

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
        pull = pull +  torch.mean(1 - cos(group, tags[-1].expand_as(group)))

    tags = torch.stack(tags)
    obj_num = len(inds)
    pull /= (obj_num + eps)

    size = (obj_num, obj_num, tag_dim)
    A = tags.unsqueeze(dim=1).expand(*size)
    B = A.permute(1, 0, 2)
    diff = torch.stack([A.unsqueeze(dim=2), B.unsqueeze(dim=2)], dim = 2)
    diff = diff.reshape(-1, 2, tag_dim)
    diff = 1 + cos(diff[:,0], diff[:,1])
    push = torch.sum(diff) - obj_num * 2
    push /= (((obj_num - 1) * obj_num + eps) * 2)

    # print('push', push, '\npull', pull)
    loss = push_weight * push + pull_weight * pull
    return loss
# def single_tag_loss(pred, gt_inds, pull_weight, push_weight):
#     assert len(pred) == len(gt_inds)
#     assert pred.numel() >= 2
#     inds = gen_group_inds(gt_inds)
#     # used for there are only negative samples
#     if len(inds) == 1:
#         return torch.mean(pred) * 0
# 
#     eps = 1e-6
#     tag_dim = pred.size(1)
#     cos = nn.CosineSimilarity(dim=1, eps=eps)
#     pull = 0
# 
#     # discard negative samples:
#     sel_inds = []
#     for ind in inds:
#         if gt_inds[ind[0]] == -1: # when it is negative
#             continue
#         sel_inds.append(ind)
#     inds = sel_inds
# 
#     for ind in inds:
#         group = pred[ind]
#         cnt = group.size(0)
#         size = (cnt, cnt, tag_dim)
#         A = group.unsqueeze(dim=1).expand(*size).unsqueeze(dim=2)
#         B = group.unsqueeze(dim=0).expand(*size).unsqueeze(dim=2)
#         simi = torch.stack([A, B], dim = 2).reshape(-1, 2, tag_dim)
#         simi = 1 - cos(simi[:,0], simi[:,1]) # cos in [-1, 1], 1 means similar
#         pull += torch.sum(simi) / (cnt * (cnt - 1) + eps)
#     obj_num = len(inds)
#     pull /= (obj_num + eps)
# 
#     push = 0
#     if obj_num > 1:
#         for idx1, ind1 in enumerate(inds):
#             for idx2, ind2 in enumerate(inds):
#                 if idx1 == idx2:
#                     continue
#                 group1 = pred[ind1]
#                 group2 = pred[ind2]
#                 cnt1 = group1.size(0)
#                 cnt2 = group2.size(0)
#                 size = (cnt1, cnt2, tag_dim)
#                 A = group1.unsqueeze(dim=1).expand(*size).unsqueeze(dim=2)
#                 B = group2.unsqueeze(dim=0).expand(*size).unsqueeze(dim=2)
#                 diff = torch.stack([A, B], dim = 2).reshape(-1, 2, tag_dim)
#                 diff = 1 + cos(diff[:,0], diff[:,1]) # cos in [-1, 1], 1 means similar
#                 push += torch.sum(diff) / (cnt1 * cnt2 + eps)
#         push /= (obj_num * (obj_num - 1) + eps)
#     print('push', push, '\npull', pull)
#     loss = push_weight * push + pull_weight * pull
#     return loss


@LOSSES.register_module
class TagCosineLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, group_anchor=True,
                        pull_weight = 1, push_weight = 1):
        super(TagCosineLoss, self).__init__()
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
        loss_tag = self.loss_weight * tag_cosine_loss(
            pred,
            (gt_inds, anchor_inds, self.group_anchor, 
             self.pull_weight, self.push_weight),
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_tag
