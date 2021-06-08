import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES
from .geometry import bbox_overlaps
import numpy as np

def final_nms_loss(gt_inds, anchor_gt_inds, gt_bboxes, proposal_list, pull_weight, push_weight, nms_thr, use_score, add_gt, pull_relax, push_relax, push_select, fix_push_score, fix_push_reg, fix_pull_score, fix_pull_reg):
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    push_loss = 0
    pull_loss = 0
    for (img_gt_inds, img_anchor_gt_inds, img_gt_bboxes, img_proposals) in zip(gt_inds, anchor_gt_inds, gt_bboxes, proposal_list):
        single_img_loss = single_nms_loss(img_gt_inds, img_anchor_gt_inds, img_gt_bboxes,img_proposals, nms_thr, use_score, add_gt, pull_relax, push_relax, push_select, fix_push_score, fix_push_reg, fix_pull_score, fix_pull_reg)
        push_loss = push_loss + single_img_loss['nms_push_loss']
        pull_loss = pull_loss + single_img_loss['nms_pull_loss']
    push_loss = push_loss / img_num
    pull_loss = pull_loss / img_num
    return {'nms_push_loss':push_loss * push_weight, 'nms_pull_loss':pull_loss * pull_weight}


def single_nms_loss(gt_inds, anchor_gt_inds, gt_box, proposals, nms_thr, use_score, add_gt, pull_relax, push_relax, push_select, fix_push_score, fix_push_reg, fix_pull_score, fix_pull_reg):
    # print(torch.sum(gt_inds - anchor_gt_inds))
    # use anchor_gt_inds instead of gt_inds
    gt_inds = anchor_gt_inds


    eps = 1e-6
    tmp_zero  = torch.mean(proposals).float() * 0 # used for return zero
    total_pull_loss = 0
    total_push_loss = 0
    pull_cnt = 0
    push_cnt = 0

    # discard negative proposals
    # print(gt_inds)
    pos_mask = gt_inds >= 0 # -2:ignore, -1:negative
    if torch.sum(pos_mask) <= 1: # when there is no positive or only one
        return {'nms_push_loss':tmp_zero, 'nms_pull_loss':tmp_zero} # return 0
    gt_inds = gt_inds[pos_mask]
    proposals = proposals[pos_mask]

    # add gt here
    if add_gt:
        gt_num = len(gt_box)
        gt_score = proposals.new_tensor([1.0] * gt_num).unsqueeze(-1).float()
        gt_proposals = torch.cat([gt_box, gt_score], dim = 1)
        add_gt_inds = proposals.new_tensor(np.arange(gt_num)).long()
        proposals = torch.cat([proposals, gt_proposals], dim = 0)
        gt_inds = torch.cat([gt_inds, add_gt_inds], dim = 0)

    # perform nms
    scores = proposals[:, 4]
    v, idx = scores.sort(0)  # sort in ascending order
    if not push_select:
        iou = bbox_overlaps(proposals[:,:4], proposals[:,:4])
    else:
        # pay attention here
        # every col has gradient for the col index proposal
        # every row doesn`t have gradient for the row index proposal
        no_gradient_proposals = proposals.detach()
        iou = bbox_overlaps(no_gradient_proposals[:,:4], proposals[:,:4])
    

    gt_iou = bbox_overlaps(gt_box, gt_box)
    max_score_box_rec = dict()
    while idx.numel() > 0:
        # print(idx)
        i = idx[-1]  # index of current largest val
        idx = idx[:-1]  # remove kept element from view
        # cacu pull loss:
        i_gt_inds = gt_inds[i]
        # print('i_gt_inds', i_gt_inds)
        i_gt_inds_value = i_gt_inds.item()
        if i_gt_inds_value in max_score_box_rec.keys():
            max_score_idx = max_score_box_rec[i_gt_inds_value]
            max_s_iou = iou[max_score_idx][i].clamp(min=eps)
            if not pull_relax:
                pull_loss = -(max_s_iou).log()
            else:
                pull_loss = -(1 - nms_thr + max_s_iou).log()
            if fix_pull_reg:
                pull_loss = pull_loss.detach()
            if use_score:
                if fix_pull_score:
                    pull_loss = pull_loss * proposals[i,4].detach()
                else:
                    pull_loss = pull_loss * proposals[i,4]
            pull_cnt += 1
        else:
            max_score_box_rec[i_gt_inds_value] = i
            pull_loss = tmp_zero
        # print(max_score_box_rec)
        if len(idx) == 0:
            break
        cur_iou = iou[i][idx]
        overlap_idx = cur_iou > nms_thr
        overlap_cur_iou = cur_iou[overlap_idx]
        overlap_idx_idx = idx[overlap_idx]
        cur_gt_inds = gt_inds[overlap_idx_idx]
        # print('cur_gt_inds', cur_gt_inds)
        # print('i_centre', [(proposals[i,1] + proposals[i,3]) * 0.5, (proposals[i,0] + proposals[i,2]) * 0.5])
        # print('cur_centre', [(proposals[overlap_idx_idx,1] + proposals[overlap_idx_idx,3]) * 0.5, (proposals[overlap_idx_idx,0] + proposals[overlap_idx_idx,2]) * 0.5])
        cur_scores = scores[overlap_idx_idx]
        if fix_push_score:
            cur_scores = cur_scores.detach()
        # cacu push loss
        push_mask = cur_gt_inds != i_gt_inds
        # check if 0
        if torch.sum(push_mask) != 0:
            cur_gt_iou = gt_iou[i_gt_inds][cur_gt_inds]
            if not push_relax:
                push_loss = -(1 - overlap_cur_iou).log()
            else:
                push_loss = -(1 + nms_thr - overlap_cur_iou).log()
            if fix_push_reg:
                push_loss = push_loss.detach()
            # push_loss = overlap_cur_iou
            if use_score:
                    push_loss = push_loss * cur_scores
            push_mask = push_mask & (overlap_cur_iou > cur_gt_iou)
            push_loss = push_loss[push_mask]
            if torch.sum(push_mask) != 0:
                push_loss = torch.mean(push_loss)
                push_cnt += int(torch.sum(push_mask).data)
            else:
                push_loss = tmp_zero
        else:
            push_loss = tmp_zero
        # print('pull_loss', pull_loss)
        # print('push_loss', push_loss)
        total_pull_loss = total_pull_loss + pull_loss
        total_push_loss = total_push_loss + push_loss
        # remove idx
        idx = idx[~overlap_idx]
   
    pull = total_pull_loss / (pull_cnt + eps)
    push = total_push_loss / (push_cnt + eps)
    return {'nms_push_loss':push, 'nms_pull_loss':pull}

@LOSSES.register_module
class FinalNMSLoss(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0, use_score = True, add_gt = False, pull_relax = True, push_relax = False, push_select = True, fix_push_score = False, fix_push_reg = False, fix_pull_score=False, fix_pull_reg=False, pull_weight = 1, push_weight = 1, nms_thr = 0.5):
        super(FinalNMSLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.nms_thr = nms_thr
        self.use_score = use_score
        self.add_gt = add_gt
        self.pull_relax = pull_relax
        self.push_relax = push_relax
        self.push_select = push_select
        self.fix_push_score = fix_push_score
        self.fix_push_reg = fix_push_reg
        self.fix_pull_score = fix_pull_score
        self.fix_pull_reg = fix_pull_reg

    def forward(self,
                gt_inds,
                anchor_gt_inds,
                gt_bboxes,
                proposal_list,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert weight is None or weight == 1, 'please use pull/push_weight instead'
        loss_nms = final_nms_loss(
            gt_inds, anchor_gt_inds, gt_bboxes, proposal_list,
            self.pull_weight, self.push_weight,
            self.nms_thr, self.use_score, self.add_gt,
            self.pull_relax, self.push_relax, self.push_select,
            self.fix_push_score, self.fix_push_reg, 
            self.fix_pull_score, self.fix_pull_reg,
            **kwargs)
        return loss_nms
