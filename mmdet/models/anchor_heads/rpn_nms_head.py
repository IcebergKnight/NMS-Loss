import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (delta2bbox, anchor_target, multi_apply)
from mmdet.ops import nms, tag_nms, soft_tag_nms
from mmdet.core import MaxIoUAssigner
from ..losses import TagOffsetLoss, SimpleTagOffsetReguLoss
from .anchor_head import AnchorHead
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class RPN_NMS_Head(AnchorHead):

    def __init__(self, in_channels, nms_loss,**kwargs):
        self.assigner = MaxIoUAssigner(pos_iou_thr = 0.5, neg_iou_thr = 0.5,
            min_pos_iou = 0.5, ignore_iof_thr = 0.5)
        super(RPN_NMS_Head, self).__init__(2, in_channels, **kwargs)
        self.nms_loss = build_loss(nms_loss)

    def _init_layers(self):
        cls_channels = self.num_anchors * self.cls_out_channels
        reg_channels = self.num_anchors * 4

        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, cls_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, reg_channels, 1)

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
             proposal_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        gt_labels = None
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list, anchor_index_list = self.get_anchors(
            featmap_sizes, img_metas)
        all_proposal_list = self.get_proposals(anchor_list, bbox_preds, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            anchor_index_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            proposal_list = all_proposal_list)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, all_bbox_gt_inds, all_bbox_anchor_inds) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        all_loss = dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

        # caculate nms loss
        # trans level based predition to img based form
        assert hasattr(self, 'rpn_anchor_list'), 'Call get_bboxes() first before loss()'
        if len(proposal_list) == 0:
            all_loss['nms_push_loss'] = 0 * cls_scores[0].mean()
            all_loss['nms_pull_loss'] = 0 * cls_scores[0].mean()
            return all_loss
            
        imgs_gt_inds = []
        imgs_anchor_gt_inds = []
        img_num = bbox_preds[0].size()[0]
        for img_idx in range(img_num):
            proposals = proposal_list[img_idx]
            proposals_anchor = self.rpn_anchor_list[img_idx]
            # assign gt for each proposals
            cur_gt_box = gt_bboxes[img_idx]
            cur_gt_box_ignore = gt_bboxes_ignore[img_idx] if gt_bboxes_ignore is not None else None
            assign_results = self.assigner.assign(proposals, cur_gt_box, cur_gt_box_ignore)
            imgs_gt_inds.append(assign_results.gt_inds - 1) # ori: -1:ignore; 0:neg; pos num: pos;   after -1: <0:ignore; >=0:pos
            anchor_assign_results = self.assigner.assign(proposals_anchor, cur_gt_box, cur_gt_box_ignore)
            imgs_anchor_gt_inds.append(anchor_assign_results.gt_inds - 1) # ori: -1:ignore; 0:neg; pos num: pos;   after -1: <0:ignore; >=0:pos
        losses_nms = self.nms_loss(imgs_gt_inds, imgs_anchor_gt_inds, gt_bboxes, proposal_list)
        if isinstance(losses_nms, dict):
            all_loss.update(losses_nms)
        else:
            all_loss['nms_loss'] = losses_nms
        return all_loss 

    def get_proposals(self, anchor_list, bbox_preds, img_metas):
        num_img = len(anchor_list)
        num_levels = len(anchor_list[0])
        propol_list = []
        for i in range(num_img):
            img_proposal_list = []
            for j in range(num_levels):
                proposals = delta2bbox(anchor_list[i][j], 
                    bbox_preds[j][i].permute(1, 2, 0).reshape(-1, 4), 
                    self.target_means, self.target_stds, 
                    img_metas[i]['img_shape'])
                img_proposal_list.append(proposals)
            propol_list.append(img_proposal_list)
        return propol_list

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg, requires_grad = False,
                   rescale=False,):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        mlvl_anchors=[]
        for i in range(num_levels):
            anchors, anchors_index=self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],self.anchor_strides[i])
            mlvl_anchors.append(anchors)
        
        result_list = []
        self.rpn_scores_list=[]
        self.rpn_anchor_list=[]
        for img_id in range(len(img_metas)):
            if requires_grad:
                cls_score_list = [
                    cls_scores[i][img_id] for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id] for i in range(num_levels)
                ]
            else:
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(num_levels)
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(num_levels)
                ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals, rpn_scores, rpn_anchor = self.get_bboxes_single(cls_score_list, 
                                               bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            if proposals is None or len(proposals) == 0:
                continue
            result_list.append(proposals)
            self.rpn_scores_list.append(rpn_scores)
            self.rpn_anchor_list.append(rpn_anchor)
        return result_list 

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
        mlvl_proposals_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
                rpn_cls_score = torch.unsqueeze(rpn_cls_score,1)
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]

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
                anchors = anchors[valid_inds, :]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            if proposals.dim() == 1:
                continue
            if cfg.nms_thr > 0 and cfg.nms_thr < 1:
                proposals, nms_idx = nms(proposals, cfg.nms_thr)
                rpn_cls_score = rpn_cls_score[nms_idx, :]
                anchors = anchors[nms_idx, :]
            proposals = proposals[:cfg.nms_post, :]
            rpn_cls_score = rpn_cls_score[:cfg.nms_post, :]
            anchors = anchors[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
            mlvl_rpn_scores.append(rpn_cls_score)
            mlvl_proposals_anchors.append(anchors)
        if len(mlvl_proposals) == 0:
            return None, None, None
        proposals = torch.cat(mlvl_proposals, 0)
        rpn_scores = torch.cat(mlvl_rpn_scores, 0)
        rpn_anchor = torch.cat(mlvl_proposals_anchors, 0)
        if cfg.nms_across_levels:
            if cfg.nms_thr > 0 and cfg.nms_thr < 1:
                proposals, nms_idx = nms(proposals, cfg.nms_thr)
                rpn_scores = rpn_scores[nms_idx, :]
                rpn_anchor = rpn_anchor[nms_idx, :]
            proposals = proposals[:cfg.max_num, :]
            rpn_scores = rpn_scores[:cfg.max_num, :]
            rpn_anchor = rpn_anchor[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
            rpn_scores = rpn_scores[topk_inds, :]
            rpn_anchor = rpn_anchor[topk_inds, :]
        self.rpn_score = rpn_scores
        self.rpn_anchor = rpn_anchor
        proposals = proposals[:, :5]
        return proposals, rpn_scores, rpn_anchor
