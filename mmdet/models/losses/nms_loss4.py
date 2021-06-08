import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES
from .geometry import bbox_overlaps
import numpy as np

def nms_loss4(gt_inds, anchor_gt_inds, gt_bboxes, proposal_list, pull_weight, push_weight, nms_thr, push_select, fix_pull_reg, min_height):
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    push_loss = 0
    pull_loss = 0
    for (img_gt_inds, img_anchor_gt_inds, img_gt_bboxes, img_proposals) in zip(gt_inds, anchor_gt_inds, gt_bboxes, proposal_list):
        single_img_loss = single_nms_loss(img_gt_inds, img_anchor_gt_inds, img_gt_bboxes,img_proposals, nms_thr, push_select, fix_pull_reg, min_height)
        push_loss = push_loss + single_img_loss['nms_push_loss']
        pull_loss = pull_loss + single_img_loss['nms_pull_loss']
    push_loss = push_loss / img_num
    pull_loss = pull_loss / img_num
    return {'nms_push_loss':push_loss * push_weight, 'nms_pull_loss':pull_loss * pull_weight}

def single_nms_loss(gt_inds, anchor_gt_inds, gt_box, proposals, nms_thr, push_select, fix_pull_reg, min_height):
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
    gt_proposal_iou = bbox_overlaps(gt_box, proposals[:,:4])
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
            # max_score_idx = max_score_box_rec[i_gt_inds_value]
            # max_s_iou = iou[max_score_idx][i].clamp(min=eps)
            max_s_iou = gt_proposal_iou[i_gt_inds][i].clamp(min=eps)
            pull_loss = -(1 - nms_thr + max_s_iou).clamp(max=1).log()
            if fix_pull_reg:
                pull_loss = pull_loss.detach()
            pull_loss = pull_loss * proposals[i,4].detach() # multipul score
            pull_cnt += 1
        else:
            max_score_box_rec[i_gt_inds_value] = i
            pull_loss = tmp_zero
        if len(idx) == 0:
            break
        cur_iou = iou[i][idx]
        overlap_idx = cur_iou > nms_thr

        # print('pull_loss', pull_loss)
        # print('push_loss', push_loss)
        total_pull_loss = total_pull_loss + pull_loss
        # remove idx
        idx = idx[~overlap_idx]


    # caculate push loss
    push_max_score_box_rec = dict()
    for i in range(len(proposals)):
        gt_ind_value = gt_inds[i].item()
        if gt_ind_value in push_max_score_box_rec.keys():
            last_max_proposal_idx = push_max_score_box_rec[gt_ind_value]
            if proposals[last_max_proposal_idx, 4] < proposals[i, 4]:
                push_max_score_box_rec[gt_ind_value] = i
        else:
            push_max_score_box_rec[gt_ind_value] = i
    for gt_id in push_max_score_box_rec.keys():
        id_gt_box = gt_box[gt_id]
        height = id_gt_box[3] - id_gt_box[1]
        if height < min_height or gt_id in max_score_box_rec.keys():
            continue
        # get max score prediction
        max_p_idx = push_max_score_box_rec[gt_id]
        # print('iou:', gt_proposal_iou[gt_id][max_p_idx])
        # print('score',proposals[max_p_idx, 4])
        # print('height',height)
        # print('max_iou',torch.max(gt_proposal_iou[gt_id]))
        tmp_push_loss = 1 - gt_proposal_iou[gt_id][max_p_idx]
        # tmp_push_loss = -(gt_proposal_iou[gt_id][max_p_idx].clamp(min=eps)).log()
        # tmp_push_loss = tmp_push_loss * (1 - proposals[max_p_idx, 4])
        # print('loss',tmp_push_loss)
        push_cnt += 1
        total_push_loss = total_push_loss + tmp_push_loss
    if push_cnt == 0:
        total_push_loss = tmp_zero
   
    pull = total_pull_loss / (pull_cnt + eps)
    push = total_push_loss / (push_cnt + eps)
    return {'nms_push_loss':push, 'nms_pull_loss':pull}

@LOSSES.register_module
class NMSLoss4(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0, push_select = True, fix_pull_reg=False, min_height = 50, pull_weight = 1, push_weight = 1, nms_thr = 0.5):
        super(NMSLoss4, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.nms_thr = nms_thr
        self.push_select = push_select
        self.fix_pull_reg = fix_pull_reg
        self.min_height = min_height

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
        loss_nms = nms_loss4(
            gt_inds, anchor_gt_inds, gt_bboxes, proposal_list,
            self.pull_weight, self.push_weight,
            self.nms_thr,
            self.push_select, self.fix_pull_reg,
            self.min_height,
            **kwargs)
        return loss_nms
