import torch
import torch.nn as nn

from .utils import weighted_loss
from ..registry import LOSSES
from .geometry import bbox_overlaps
import numpy as np

def nms_loss3(gt_inds, anchor_gt_inds, gt_bboxes, proposal_list, pull_weight, push_weight, nms_thr, add_gt, push_select, fix_pull_reg, pull_score_up):
    assert len(gt_inds) > 0
    img_num = len(gt_inds)
    push_loss = 0
    pull_loss = 0
    for (img_gt_inds, img_anchor_gt_inds, img_gt_bboxes, img_proposals) in zip(gt_inds, anchor_gt_inds, gt_bboxes, proposal_list):
        single_img_loss = single_nms_loss(img_gt_inds, img_anchor_gt_inds, img_gt_bboxes,img_proposals, nms_thr, add_gt, push_select, fix_pull_reg, pull_score_up)
        push_loss = push_loss + single_img_loss['nms_push_loss']
        pull_loss = pull_loss + single_img_loss['nms_pull_loss']
    push_loss = push_loss / img_num
    pull_loss = pull_loss / img_num
    return {'nms_push_loss':push_loss * push_weight, 'nms_pull_loss':pull_loss * pull_weight}

def cacu_giou(box1, box2):
    eps = 1e-6

    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    xi_1 = torch.max(box1[0], box2[0])
    yi_1 = torch.max(box1[1], box2[1])
    xi_2 = torch.min(box1[2], box2[2])
    yi_2 = torch.min(box1[3], box2[3])

    xu_1 = torch.min(box1[0], box2[0])
    yu_1 = torch.min(box1[1], box2[1])
    xu_2 = torch.max(box1[2], box2[2])
    yu_2 = torch.max(box1[3], box2[3])

    wi = (xi_2 - xi_1).clamp(min=0)
    hi = (yi_2 - yi_1).clamp(min=0)
    wu = (xu_2 - xu_1).clamp(min=0)
    hu = (yu_2 - yu_1).clamp(min=0)

    au = area1 + area2 - wi * hi
    ac = wu * hu
    giou = (ac - au) / (ac + eps)
    return giou

def smooth_l1(gt_box, boxes, beta=0.1):
    gt_w = gt_box[2] - gt_box[0]
    gt_h = gt_box[3] - gt_box[1]
    boxes_w = boxes[:, 2] - boxes[:, 0]
    boxes_h = boxes[:, 3] - boxes[:, 1]
    offset_x = (gt_box[0] - boxes[:, 0]) / boxes_w
    offset_y = (gt_box[1] - boxes[:, 1]) / boxes_h
    offset_w = (gt_w / boxes_w).log()
    offset_h = (gt_h / boxes_h).log()
    diff = torch.stack([offset_x,offset_y,offset_w,offset_h], dim = 1)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss.mean(dim = -1)
def single_nms_loss(gt_inds, anchor_gt_inds, gt_box, proposals, nms_thr, add_gt, push_select, fix_pull_reg, pull_score_up):
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

    # print('before proposals\n', proposals)
    # print('before gt_inds\n', gt_inds)
    # print('before gt_box\n', gt_box)

    # add gt here
    if add_gt:
        gt_num = len(gt_box)
        gt_score = proposals.new_tensor([1.0] * gt_num).unsqueeze(-1).float()
        gt_proposals = torch.cat([gt_box, gt_score], dim = 1)
        add_gt_inds = proposals.new_tensor(np.arange(gt_num)).long()
        proposals = torch.cat([proposals, gt_proposals], dim = 0)
        gt_inds = torch.cat([gt_inds, add_gt_inds], dim = 0)


    # print('proposals\n', proposals)
    # print('gt_inds\n', gt_inds)
    # print('gt_box\n', gt_box)


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
            # max_s_giou = cacu_giou(proposals[i], proposals[max_score_idx])
            max_s_iou = iou[max_score_idx][i].clamp(min=eps)
            pull_loss = -(1 - nms_thr + max_s_iou).log()
            if pull_score_up:
                pull_loss = pull_loss - (proposals[i,4]).log()
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
        overlap_cur_iou = cur_iou[overlap_idx]
        overlap_idx_idx = idx[overlap_idx]
        cur_gt_inds = gt_inds[overlap_idx_idx]
        cur_scores = scores[overlap_idx_idx]
        # cacu push loss
        push_mask = cur_gt_inds != i_gt_inds
        # check if 0
        if torch.sum(push_mask) != 0:
            cur_gt_iou = gt_iou[i_gt_inds][cur_gt_inds]
            push_loss = -(1 + nms_thr - overlap_cur_iou).log() - (cur_scores).log()
            push_loss = push_loss * cur_scores.detach()
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
class NMSLoss3(nn.Module):

    def __init__(self, reduction='none', loss_weight=1.0, add_gt = False, push_select = True, fix_pull_reg=False, pull_score_up = False, pull_weight = 1, push_weight = 1, nms_thr = 0.5):
        super(NMSLoss3, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.nms_thr = nms_thr
        self.add_gt = add_gt
        self.push_select = push_select
        self.fix_pull_reg = fix_pull_reg
        self.pull_score_up = pull_score_up

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
        loss_nms = nms_loss3(
            gt_inds, anchor_gt_inds, gt_bboxes, proposal_list,
            self.pull_weight, self.push_weight,
            self.nms_thr, self.add_gt,
            self.push_select,
            self.fix_pull_reg, self.pull_score_up,
            **kwargs)
        return loss_nms
