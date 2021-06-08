import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (delta2bbox, anchor_target, multi_apply, gen_map_index, offset_tag)
from mmdet.ops import nms, tag_nms
from .anchor_head import AnchorHead
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class RPN_S_TAGHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPN_S_TAGHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        cls_channels = self.num_anchors * self.cls_out_channels
        reg_channels = self.num_anchors * 4
        tag_channels = self.tag_dim

        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, cls_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, reg_channels, 1)
        self.pool = nn.MaxPool2d(4, 4)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.receptive_conv = nn.Conv2d(
            self.feat_channels, self.feat_channels, 3, padding=1)
        self.rpn_tag = nn.Conv2d(self.feat_channels * 2, tag_channels, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.rpn_tag, std=0.01)
        normal_init(self.receptive_conv, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        receptive_feat = self.receptive_conv(self.pool(x))
        receptive_feat = F.relu(receptive_feat, inplace=True)
        receptive_feat = self.upsample(receptive_feat)
        x = torch.cat((x, receptive_feat), dim = 1)
        rpn_tag_pred = self.rpn_tag(x)
        exp_size = (rpn_tag_pred.size(0), self.num_anchors, rpn_tag_pred.size(1),
                    rpn_tag_pred.size(2), rpn_tag_pred.size(3))
        target_size = (rpn_tag_pred.size(0), rpn_tag_pred.size(1) * self.num_anchors,
                    rpn_tag_pred.size(2), rpn_tag_pred.size(3))
        rpn_tag_pred = rpn_tag_pred.unsqueeze(dim=1).expand(*exp_size).contiguous().view(*target_size)
        return rpn_cls_score, rpn_bbox_pred, rpn_tag_pred


    def loss(self,
             cls_scores,
             bbox_preds,
             tag_pred,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        gt_labels = None
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list, anchor_index_list = self.get_anchors(
            featmap_sizes, img_metas)
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
            sampling=self.sampling)
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

        # caculate tag loss
        # trans level based predition to img based form
        imgs_tag = []
        img_num = tag_pred[0].size()[0]
        for img_idx in range(img_num):
            single_img_tag = torch.cat([single_level_tag[img_idx].permute(1,2,0).contiguous().view(-1, self.tag_dim) for single_level_tag in tag_pred], dim = 0)
            # print(single_img_tag.size())
            imgs_tag.append(single_img_tag)
        
        losses_tag = self.loss_tag(imgs_tag, all_bbox_gt_inds, all_bbox_anchor_inds)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox, loss_rpn_tag = losses_tag)
    
    def get_bboxes(self, cls_scores, bbox_preds, tag_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        mlvl_anchors=[]
        mlvl_anchors_index=[]
        for i in range(num_levels):
            anchors, anchors_index=self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],self.anchor_strides[i])
            mlvl_anchors.append(anchors)
            mlvl_anchors_index.append(anchors_index)
        
        result_list = []
        self.rpn_scores_list=[]
        self.rpn_tag_list=[]
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            tag_pred_list = [
                tag_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               tag_pred_list, mlvl_anchors,
                                               mlvl_anchors_index, img_shape,
                                               scale_factor, cfg, rescale)
            if proposals is None:
                continue
            result_list.append(proposals)
            if hasattr(self, 'rpn_score'):
                rpn_scores = self.get_score_single()
                self.rpn_scores_list.append(rpn_scores)
            if hasattr(self, 'rpn_tag'):
                rpn_tag = self.get_tag_single()
                self.rpn_tag_list.append(rpn_tag)
        return result_list 

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          tag_preds,
                          mlvl_anchors,
                          mlvl_anchors_index,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        mlvl_tags = []
        mlvl_rpn_scores = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            rpn_tag_pred = tag_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:] 
            anchors = mlvl_anchors[idx]
            anchors_inds = mlvl_anchors_index[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            # sel_index = gen_map_index(int(rpn_bbox_pred.size(0) / 4),
            #                           rpn_bbox_pred.size(1),
            #                           rpn_bbox_pred.size(2)) # [c, h, w, 3(index)]
            # sel_index = sel_index.permute(1, 2, 0, 3).reshape(-1, 3)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
                rpn_cls_score = torch.unsqueeze(rpn_cls_score,1)
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            ori_rpn_tag_pred = rpn_tag_pred.reshape(-1, self.tag_dim, rpn_tag_pred.size(1), rpn_tag_pred.size(2)).permute(0, 2, 3, 1)
            rpn_tag_pred = rpn_tag_pred.permute(1, 2, 0).reshape(-1, self.tag_dim)
            if 'min_score' in cfg.keys() and torch.min(scores) < cfg.min_score:
                sele_inds = scores > cfg.min_score
                rpn_bbox_pred = rpn_bbox_pred[sele_inds, :]
                rpn_cls_score = rpn_cls_score[sele_inds, :]
                anchors = anchors[sele_inds, :]
                scores = scores[sele_inds]
                # sel_index = sel_index[sele_inds, :]
                rpn_tag_pred = rpn_tag_pred[sele_inds, :]
                anchors_inds = anchors_inds[sele_inds]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                rpn_tag_pred = rpn_tag_pred[topk_inds, :]
                # sel_index = sel_index[topk_inds, :]
                rpn_cls_score = rpn_cls_score[topk_inds, :]
                anchors = anchors[topk_inds, :]
                anchors_inds = anchors_inds[topk_inds]
                scores = scores[topk_inds]
            # rpn_bbox_pred = anchors.new_zeros(rpn_bbox_pred.size()) # used for no reg
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            # if True: # offset tag
            #     new_rpn_tag_pred = offset_tag(ori_rpn_tag_pred, sel_index, anchors, 
            #         rpn_bbox_pred, img_shape[0] / ori_rpn_tag_pred.size(1),
            #         self.target_means,self.target_stds)
            #     if new_rpn_tag_pred is not None:
            #         rpn_tag_pred = new_rpn_tag_pred

            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                rpn_cls_score = rpn_cls_score[valid_inds, :]
                scores = scores[valid_inds]
                rpn_tag_pred = rpn_tag_pred[valid_inds]
                anchors_inds = anchors_inds[valid_inds]

            # proposals = torch.cat([proposals, scores.unsqueeze(-1), rpn_tag_pred], dim=-1)
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            if proposals.dim() == 1:
                continue
            # proposals, nms_idx = tag_nms(proposals, rpn_tag_pred, anchors_inds)
            proposals, nms_idx = nms(proposals, cfg.nms_thr)
            rpn_cls_score = rpn_cls_score[nms_idx, :]
            rpn_tag_pred = rpn_tag_pred[nms_idx, :]
            proposals = proposals[:cfg.nms_post, :]
            rpn_cls_score = rpn_cls_score[:cfg.nms_post, :]
            rpn_tag_pred = rpn_tag_pred[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
            mlvl_tags.append(rpn_tag_pred)
            mlvl_rpn_scores.append(rpn_cls_score)
        if len(mlvl_proposals) == 0:
            return None
        proposals = torch.cat(mlvl_proposals, 0)
        tags = torch.cat(mlvl_tags, 0)
        rpn_scores = torch.cat(mlvl_rpn_scores, 0)
        if cfg.nms_across_levels:
            proposals, nms_idx = nms(proposals, cfg.nms_thr)
            rpn_scores = rpn_scores[nms_idx, :]
            tags = tags[nms_idx, :]
            proposals = proposals[:cfg.max_num, :]
            rpn_scores = rpn_scores[:cfg.max_num, :]
            tags = tags[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
            rpn_scores = rpn_scores[topk_inds, :]
            tags = tags[topk_inds, :]
        self.rpn_score = rpn_scores
        self.pred_tag = tags
        proposals = proposals[:, :5]
        return proposals

    # this method must be called after 'get_boxes_single()'
    def get_score_single(self):
        if hasattr(self, 'rpn_score') and self.rpn_score is not None:
            return self.rpn_score
        else:
            return None
    # this method must be called after 'get_boxes_single()'
    def get_tag_single(self):
        if hasattr(self, 'pred_tag') and self.pred_tag is not None:
            return self.pred_tag
        else:
            return None

