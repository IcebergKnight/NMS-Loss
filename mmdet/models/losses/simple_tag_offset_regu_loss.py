import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES
from .geometry import bbox_overlaps
import numpy as np

def gen_group_inds(gt_inds):
    target = gt_inds.cpu().numpy()
    gt_label = np.unique(target)
    res = []
    for i in gt_label:
        res.append((np.argwhere(target == i).reshape(-1)))
    return res



def simple_tag_offset_regu_loss(pred, extras):
    (gt_inds, gt_bboxes, pull_weight, push_weight, regu_thr, w_low, w_high, expand, iou_thr) = extras
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    push_loss = 0
    pull_loss = 0
    for (img_pred, img_gt_inds, img_gt_bboxes) in zip(pred, gt_inds, gt_bboxes):
        single_img_loss = single_tag_loss(img_pred, img_gt_inds, img_gt_bboxes,
                                          regu_thr, w_low, w_high, expand, iou_thr)
        push_loss = push_loss + single_img_loss['push_loss']
        pull_loss = pull_loss + single_img_loss['pull_loss']
    push_loss = push_loss / img_num
    pull_loss = pull_loss / img_num
    return {'push_loss':push_loss * push_weight, 'pull_loss':pull_loss * pull_weight}

def get_weight(boxes, gt_order, expand = 0.25, iou_thr = 0, 
                    w_low = 1, w_high = 10):
    height = boxes[:, 3] - boxes[:, 1] + 1
    width = boxes[:, 2] - boxes[:, 0] + 1
    exp_h = height * expand
    exp_w = width * expand
    boxes[:, 0] -= exp_w
    boxes[:, 2] += exp_w
    boxes[:, 1] -= exp_h
    boxes[:, 3] += exp_h
    iou = bbox_overlaps(boxes, boxes)
    # print(iou)
    # print(gt_order)
    push_weight = torch.zeros_like(iou, device=iou.device)
    pull_weight = torch.zeros_like(iou[0], device=iou.device)

    push_weight[iou == 0] = w_low
    push_weight[iou > 0] = w_high
    push_weight = torch.cat([push_weight[i].unsqueeze(0) for i in gt_order], dim = 0)
    push_weight = torch.cat([push_weight[:,i].unsqueeze(1) for i in gt_order], dim = 1)
    
    iou_sum = torch.sum(iou, dim = 1)
    pull_weight[iou_sum == 0] = w_high
    pull_weight[iou_sum > 0] = w_low
    pull_weight = [pull_weight[i] for i in gt_order]
    return push_weight, pull_weight

def single_tag_loss(pred, gt_inds, gt_box,
    regu_thr = 1e-3, w_low = 0.1, w_high = 10, expand = 0.25, iou_thr = 0):
    # some super paramater
    # regu_thr = 1e-3
    # w_low = 0.1
    # w_high = 10
    # expand = 0.25
    # iou_thr = 0

    eps = 1e-6
    tmp_zero  = torch.mean(pred).float() * 0 # used for return zero
    tags = []
    pull = 0

    inds = gen_group_inds(gt_inds)
    # discard negative samples:
    sel_inds = []
    gts = []
    for ind in inds:
        gt_id = gt_inds[ind[0]]
        if gt_id < 0: # when it is negative or ignore
            continue
        sel_inds.append(ind)
        gts.append(gt_id)
    inds = sel_inds
    gt_order = np.array(gts)

    # used for there are only negative samples
    if len(inds) == 0:
        return {'push_loss':tmp_zero, 'pull_loss':tmp_zero} # return 0
    
    item_push_weight, item_pull_weight = get_weight(gt_box, gt_order, 
            expand = expand, iou_thr = iou_thr, w_low = w_low, w_high = w_high)

    for i, ind in enumerate(inds):
        group = pred[ind]
        tags.append(torch.mean(group, dim=0))
        cur_pull = torch.mean((group - tags[-1].expand_as(group))**2, dim=1)
        regu_idx = cur_pull<regu_thr
        cur_pull[regu_idx] = cur_pull[regu_idx] * 0 # regu
        pull = pull +  torch.mean(cur_pull) * item_pull_weight[i]

    tags = torch.stack(tags)
    num = tags.size()[0]
    size = (num, num, tags.size()[1])
    A = tags.unsqueeze(dim=1).expand(*size)
    B = A.permute(1, 0, 2)
    diff = A - B
    diff = torch.pow(diff, 2).mean(dim=2)
    # print(diff)
    push = torch.exp(-diff)
    regu_idx = push<regu_thr
    item_push_weight[regu_idx] = 0 # regu
    push = push * item_push_weight
    push = torch.sum(push) - num * w_high
    pull = pull/(num + eps)
    push = push/((num - 1) * num + eps) * 0.5
    return {'push_loss':push, 'pull_loss':pull}


@LOSSES.register_module
class SimpleTagOffsetReguLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, group_anchor=True, 
                pull_weight = 1, push_weight = 1, regu_thr = 1e-3, 
                w_low = 0.1, w_high = 10, expand = 0.25, iou_thr = 0):
        super(SimpleTagOffsetReguLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.group_anchor = group_anchor
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regu_thr = regu_thr
        self.w_low = w_low
        self.w_high = w_high
        self.expand = expand
        self.iou_thr = iou_thr

    def forward(self,
                pred,
                gt_inds,
                gt_bboxes,
                proposal_list = None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert weight is None or weight == 1, 'please use pull/push_weight instead'
        loss_tag = simple_tag_offset_regu_loss(
            pred,
            (gt_inds, gt_bboxes,
            self.pull_weight, self.push_weight,
            self.regu_thr, self.w_low, self.w_high, 
            self.expand, self.iou_thr),
            **kwargs)
        return loss_tag
