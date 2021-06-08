import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
import math

from mmdet.core import delta2bbox
from mmdet.ops import nms, soft_nms
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)
        self.neck_mask = None

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])


    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        mlvl_rpn_scores = []
        for idx in range(len(cls_scores)):
            # if idx in [3]:
            #     continue   
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            feat_h = rpn_cls_score.size(0)
            anchor_n = rpn_cls_score.size(2) / (1 if self.use_sigmoid_cls else 2)
            # added by joeyfang:
            # use a mask to avoid predictions from anchors outside the img
            # inside_masks = []
            # base_anchors = self.anchor_generators[idx].gen_base_anchors()
            # for base_anchor in base_anchors:
            #     feat_h = rpn_cls_score.size(0)
            #     feat_w = rpn_cls_score.size(1)
            #     anchor_h = base_anchor[3] - base_anchor[1]
            #     anchor_w = base_anchor[2] - base_anchor[0]
            #     pad_x = int(anchor_w / 2)
            #     pad_y = int(anchor_h / 2)
            #     inside_mask = rpn_cls_score.new_zeros(rpn_cls_score.size()[:2])
            #     if not (pad_x >= feat_w / 2 or pad_y >= feat_h / 2):
            #         inside_mask[pad_y: feat_h - pad_y + 1, 
            #                     pad_x: feat_w - pad_x + 1] = 1
            #     inside_masks.append(inside_mask.unsqueeze(2))
            # inside_masks = torch.cat(inside_masks, dim=2)

            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
                rpn_cls_score = torch.unsqueeze(rpn_cls_score,1)
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            # scores *= inside_masks.reshape(-1)

            if self.neck_mask is not None:
                s_mask = self.neck_mask[0]
                s_mask = F.upsample(s_mask, scale_factor = feat_h / s_mask.size(2)).squeeze()
                s_mask = s_mask.unsqueeze(-1)
                s_mask = torch.cat((s_mask, s_mask), dim = -1)
                s_mask = s_mask.reshape(-1)
                s_mask = s_mask.sigmoid()
                scores *= s_mask

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if 'min_score' in cfg.keys() and torch.min(scores) < cfg.min_score:
                sele_inds = scores > cfg.min_score
                rpn_bbox_pred = rpn_bbox_pred[sele_inds, :]
                rpn_cls_score = rpn_cls_score[sele_inds, :]
                anchors = anchors[sele_inds, :]
                scores = scores[sele_inds]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                rpn_cls_score = rpn_cls_score[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                rpn_cls_score = rpn_cls_score[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            if proposals.dim() == 1:
                continue
            # proposals, nms_idx = soft_nms(proposals, cfg.nms_thr, min_score=cfg.min_score)
            proposals, nms_idx = nms(proposals, cfg.nms_thr)
            rpn_cls_score = rpn_cls_score[nms_idx, :]
            # print( proposals.size())
            proposals = proposals[:cfg.nms_post, :]
            rpn_cls_score = rpn_cls_score[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
            mlvl_rpn_scores.append(rpn_cls_score)
        if len(mlvl_proposals) == 0:
            return None
        proposals = torch.cat(mlvl_proposals, 0)
        rpn_scores = torch.cat(mlvl_rpn_scores, 0)
        if cfg.nms_across_levels:
            proposals, nms_idx = nms(proposals, cfg.nms_thr)
            rpn_scores = rpn_scores[nms_idx, :]
            proposals = proposals[:cfg.max_num, :]
            rpn_scores = rpn_scores[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
            rpn_scores = rpn_scores[topk_inds, :]
        self.rpn_score = rpn_scores
        proposals = proposals[:, :5]
        return proposals

    # this method must be called after 'get_boxes_single()'
    def get_score_single(self):
        if hasattr(self, 'rpn_score') and self.rpn_score is not None:
            return self.rpn_score
        else:
            return None
