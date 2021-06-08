import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES
from .geometry import bbox_overlaps
import numpy as np

# @weighted_loss
def end2end_tag_loss(pred, extras):
    (gt_inds, gt_bboxes, proposal_list, pull_weight, push_weight, tag_thr, min_height) = extras
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    push_loss = 0
    pull_loss = 0
    for (img_pred, img_gt_inds, img_gt_bboxes, img_proposals) in zip(pred, gt_inds, gt_bboxes, proposal_list):
        single_img_loss = single_tag_loss(img_pred, img_gt_inds, img_gt_bboxes,
                                    img_proposals, tag_thr, min_height)
        push_loss = push_loss + single_img_loss['push_loss']
        pull_loss = pull_loss + single_img_loss['pull_loss']
    push_loss = push_loss / img_num
    pull_loss = pull_loss / img_num
    return {'push_loss':push_loss * push_weight, 'pull_loss':pull_loss * pull_weight}

def single_tag_loss(pred, gt_inds, gt_box, proposals, tag_thr, min_height):
    eps = 1e-6
    tmp_zero  = torch.mean(pred).float() * 0 # used for return zero
    total_pull_loss = 0
    total_push_loss = 0
    pull_cnt = 0
    push_cnt = 0


    # discard negative proposals
    pos_mask = gt_inds >= 0 # -2:ignore, -1:negative
    if torch.sum(pos_mask) <= 1: # when there is no positive or only one
        return {'push_loss':tmp_zero, 'pull_loss':tmp_zero} # return 0
    pred = pred[pos_mask]
    gt_inds = gt_inds[pos_mask]
    proposals = proposals[pos_mask]

    # discard too small proposals
    heights = proposals[:,3] - proposals[:,1]
    height_mask = heights > min_height
    if torch.sum(height_mask) <= 1: # when there is no positive or only one
        return {'push_loss':tmp_zero, 'pull_loss':tmp_zero} # return 0
    pred = pred[height_mask]
    gt_inds = gt_inds[height_mask]
    proposals = proposals[height_mask]
    # perform nms

    scores = proposals[:, 4]
    v, idx = scores.sort(0)  # sort in ascending order
    iou = bbox_overlaps(proposals[:,:4], proposals[:,:4]) # to check wether has overlap
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        idx = idx[:-1]  # remove kept element from view
        if len(idx) == 0:
            break
        cur_iou = iou[i].clone()
        cur_iou = cur_iou[idx]
        overlap_idx = cur_iou>0
        overlap_idx_idx = idx[overlap_idx]
        i_gt_inds = gt_inds[i]
        #print('i_gt_inds', i_gt_inds)
        cur_gt_inds = gt_inds[overlap_idx_idx]
        # print('cur_gt_inds', cur_gt_inds)
        # print('i_centre', [(proposals[i,1] + proposals[i,3]) * 0.5, (proposals[i,0] + proposals[i,2]) * 0.5])
        # print('cur_centre', [(proposals[overlap_idx_idx,1] + proposals[overlap_idx_idx,3]) * 0.5, (proposals[overlap_idx_idx,0] + proposals[overlap_idx_idx,2]) * 0.5])
        cur_scores = scores[overlap_idx_idx]
        pull_mask = cur_gt_inds == i_gt_inds
        push_mask = cur_gt_inds != i_gt_inds
        i_tag = pred[i]
        cur_tags = pred[overlap_idx_idx]
        diff = torch.mean((cur_tags - i_tag.expand_as(cur_tags))**2, dim=1)
        # print('diff',diff)
        check_diff = diff < tag_thr
        pull_mask = pull_mask & (~check_diff)
        push_mask = push_mask & check_diff
        pull_loss = tmp_zero 
        # check if 0
        if torch.sum(pull_mask) != 0:
            pull_loss = torch.mean(diff[pull_mask] * cur_scores[pull_mask])
            pull_cnt += 1
        push_loss = tmp_zero 
        # check if 0
        if torch.sum(push_mask) != 0:
            push_loss = torch.mean(torch.exp(-diff[push_mask]) * cur_scores[push_mask])
            push_cnt += 1
        # print('pull_loss', pull_loss)
        # print('push_loss', push_loss)
        total_pull_loss = total_pull_loss + pull_loss
        total_push_loss = total_push_loss + push_loss
        # remove idx
        remove_idx = overlap_idx_idx[check_diff]
        new_idx = []
        for id in idx:
            if id not in remove_idx:
                new_idx.append(id)
        idx = torch.LongTensor(new_idx)
   
    pull = total_pull_loss / (pull_cnt + eps)
    push = total_push_loss / (push_cnt + eps)
    return {'push_loss':push, 'pull_loss':pull}


@LOSSES.register_module
class End2EndTagLoss(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0, group_anchor=False, 
                pull_weight = 1, push_weight = 1, tag_thr = 0.1, min_height = 0):
        super(End2EndTagLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.group_anchor = group_anchor
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.tag_thr = tag_thr
        self.min_height = min_height

    def forward(self,
                pred,
                gt_inds,
                gt_bboxes,
                proposal_list,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert weight is None or weight == 1, 'please use pull/push_weight instead'
        loss_tag = end2end_tag_loss(
            pred,
            (gt_inds, gt_bboxes, proposal_list,
            self.pull_weight, self.push_weight,
            self.tag_thr, self.min_height),
            **kwargs)
        return loss_tag
