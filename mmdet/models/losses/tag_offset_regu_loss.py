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



def tag_offset_regu_loss(pred, extras):
    (gt_inds, anchor_inds, group_anchor, pull_weight, 
        push_weight, anchor_list, offset_preds, get_trans_center, gt_bboxes) = extras
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    push_loss = 0
    pull_loss = 0

    if anchor_inds is None:
        anchor_inds = [None] * img_num
    for (img_pred, img_gt_inds, img_anchor_inds, offset_pred, gt_box) in zip(pred, gt_inds, anchor_inds, offset_preds, gt_bboxes):
        single_img_loss = single_tag_loss(img_pred, img_gt_inds, anchor_list, offset_pred, get_trans_center, gt_box)
        push_loss = push_loss + single_img_loss['push_loss']
        pull_loss = pull_loss + single_img_loss['pull_loss']
    push_loss = push_loss / img_num
    pull_loss = pull_loss / img_num
    return {'push_loss':push_loss * push_weight, 'pull_loss':pull_loss * pull_weight}

def get_push_weight(boxes, gt_order, expand = 0.25, iou_thr = 0, 
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
    weight = torch.zeros_like(iou, device=iou.device)
    weight[iou == 0] = w_low
    weight[iou > 0] = w_high
    weight = torch.cat([weight[i].unsqueeze(0) for i in gt_order], dim = 0)
    weight = torch.cat([weight[:,i].unsqueeze(1) for i in gt_order], dim = 1)
    return weight

def single_tag_loss(pred, gt_inds,
                    anchor_list, offset_pred, get_trans_center, gt_box):
    regu_thr = 1e-3
    tmp_zero  = torch.mean(pred).float() * 0 # used for return zero

    inds = gen_group_inds(gt_inds)
    # used for there are only negative samples
    if len(inds) == 1:
        return {'push_loss':tmp_zero, 'pull_loss':tmp_zero} # return 0

    eps = 1e-6
    tags = []
    pull = 0

    # discard negative samples:
    sel_inds = []
    gts = []
    for ind in inds:
        gt_id = gt_inds[ind[0]]
        if gt_id == -1: # when it is negative
            continue
        sel_inds.append(ind)
        gts.append(gt_id)
    inds = sel_inds
    gt_order = np.array(gts)

    tag_dim = pred.size(0)
    pred = pred.unsqueeze(0) # [N,C,H,W]

    for ind in inds:
        centers = get_trans_center(anchor_list[ind], offset_pred[ind])
        centers = centers.reshape(1,1,centers.size(0), 2)
        group = nn.functional.grid_sample(pred, centers).reshape([tag_dim, -1]).permute(1,0)
        tags.append(torch.mean(group, dim=0))
        cur_pull = torch.mean((group - tags[-1].expand_as(group))**2, dim=1)
        regu_idx = cur_pull<regu_thr
        cur_pull[regu_idx] = cur_pull[regu_idx] * 0 # regu
        pull = pull +  torch.mean(cur_pull)

    tags = torch.stack(tags)
    num = tags.size()[0]
    size = (num, num, tags.size()[1])
    A = tags.unsqueeze(dim=1).expand(*size)
    B = A.permute(1, 0, 2)
    diff = A - B
    diff = torch.pow(diff, 2).mean(dim=2)
    # print(diff)
    push = torch.exp(-diff)
    w_low = 0.1
    w_high = 1
    push_weight = get_push_weight(gt_box, gt_order, expand = 0.25, iou_thr = 0, 
                                w_low = w_low, w_high = w_high)
    regu_idx = push<regu_thr
    push_weight[regu_idx] = 0
    push = push * push_weight
    push = torch.sum(push) - num * w_high
    pull = pull/(num + eps)
    push = push/((num - 1) * num + eps) * 0.5
    return {'push_loss':push, 'pull_loss':pull}


@LOSSES.register_module
class TagOffsetReguLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, group_anchor=True, 
                pull_weight = 1, push_weight = 1):
        super(TagOffsetReguLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.group_anchor = group_anchor
        self.pull_weight = pull_weight
        self.push_weight = push_weight

    def forward(self,
                pred, # size: [tag_dim, height, width]
                gt_inds, # size: [27648, 1] (anchor*height*width)
                anchor_inds, # size: [27648, 1] (anchor*height*width)
                anchor_list, # size: [27648, 4] (anchor*height*width)
                offset_pred, # size: [27648, 4] (anchor*height*width) 
                get_trans_center,
                gt_bboxes,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert weight is None or weight == 1, 'please use pull/push_weight instead'
        loss_tag = tag_offset_regu_loss(
            pred,
            (gt_inds, anchor_inds, self.group_anchor, 
            self.pull_weight, self.push_weight,
            anchor_list, offset_pred,
            get_trans_center, gt_bboxes),
            **kwargs)
        return loss_tag
