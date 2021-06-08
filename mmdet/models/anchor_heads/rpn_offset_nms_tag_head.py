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
class RPN_Offset_NMS_TAGHead(AnchorHead):

    def __init__(self, in_channels, tag_anchor_free = True, **kwargs):
        assert tag_anchor_free is True, 'ONLY support tag_anchor_free'
        self.tag_anchor_free = tag_anchor_free
        self.assigner = MaxIoUAssigner(pos_iou_thr = 0.5, neg_iou_thr = 0.5,
            min_pos_iou = 0.5, ignore_iof_thr = 0.5)
        super(RPN_Offset_NMS_TAGHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        cls_channels = self.num_anchors * self.cls_out_channels
        reg_channels = self.num_anchors * 4
        tag_channels = self.tag_dim if self.tag_anchor_free else self.num_anchors * self.tag_dim

        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels, cls_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, reg_channels, 1)
        self.pool = nn.MaxPool2d(4, 4)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.receptive_conv = nn.Conv2d(
            self.feat_channels, self.feat_channels, 3, padding=1)
        self.feat_smooth = nn.Conv2d(self.feat_channels * 2, self.feat_channels, 1)
        self.tag_feat = nn.Conv2d(self.feat_channels  + cls_channels + reg_channels,
                                  self.feat_channels, 7, padding = 3)
        self.rpn_tag = nn.Conv2d(self.feat_channels, tag_channels, 1)

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
        x = self.feat_smooth(x)
        x = F.relu(x, inplace=True)
        x = torch.cat((x, rpn_cls_score, rpn_bbox_pred), dim = 1)
        x = self.tag_feat(x)
        x = F.relu(x, inplace=True)
        rpn_tag_pred = self.rpn_tag(x)
        return rpn_cls_score, rpn_bbox_pred, rpn_tag_pred


    def loss(self,
             cls_scores,
             bbox_preds,
             tag_pred,
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
        imgs_gt_inds = []
        img_num = tag_pred[0].size()[0]
        for img_idx in range(img_num):
            # WARNING!!! ONLY support single level net
            assert len(tag_pred) == 1, 'ONLY support single level net!'
            # First, get tags of each proposal
            cur_tag_map = tag_pred[0][img_idx].unsqueeze(0) # [N,C,H,W]
            proposals = proposal_list[img_idx]    
            centers = torch.cat([((proposals[:,1] + proposals[:,3]) * 0.5).unsqueeze(1), ((proposals[:,0] + proposals[:,2]) * 0.5).unsqueeze(1)], dim = 1) # [y, x]
            height = img_metas[0]['pad_shape'][0]
            width = img_metas[0]['pad_shape'][1]
            # print(height, width)
            # normalize to [-1, 1]
            centers[:,0] = centers[:,0] / (height/2) - 1
            centers[:,1] = centers[:,1] / (width/2) - 1
            centers = centers.reshape(1,1,centers.size(0), 2)
            cur_tags = nn.functional.grid_sample(cur_tag_map, centers).reshape([self.tag_dim, -1]).permute(1,0)
            imgs_tag.append(cur_tags)
            # Second, assign gt for each proposals
            cur_gt_box = gt_bboxes[img_idx]
            cur_gt_box_ignore = gt_bboxes_ignore[img_idx] if gt_bboxes_ignore is not None else None
            assign_results = self.assigner.assign(proposals, cur_gt_box, cur_gt_box_ignore)
            imgs_gt_inds.append(assign_results.gt_inds - 1) # ori: -1:ignore; 0:neg; pos num: pos;   after -1: <0:ignore; >=0:pos
        # assert isinstance(self.loss_tag, SimpleTagOffsetReguLoss), 'ONLY support SimpleTagOffsetReguLoss!'
        losses_tag = self.loss_tag(imgs_tag, imgs_gt_inds, gt_bboxes, proposal_list)
        all_loss = dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)
        if isinstance(losses_tag, dict):
            all_loss.update(losses_tag)
        else:
            all_loss['loss_rpn_tag'] = losses_tag
        return all_loss 
    def get_bboxes(self, cls_scores, bbox_preds, tag_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        assert num_levels == 1, 'Not suport multi level scale'
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
        rpn_cls_score = cls_scores[0]
        rpn_bbox_pred = bbox_preds[0]
        assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:] 
        anchors = mlvl_anchors[0]
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

        proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
        if proposals.dim() == 1:
            return None
        
        # get offset tag here
        tags = tag_preds[0].unsqueeze(0) # [N,C,H,W]
        centers = torch.cat([((proposals[:,1] + proposals[:,3]) * 0.5).unsqueeze(1), ((proposals[:,0] + proposals[:,2]) * 0.5).unsqueeze(1)], dim = 1) # [y, x]
        height = img_shape[0]
        width = img_shape[1]
        # normalize to [-1, 1]
        centers[:,0] = centers[:,0] / (height/2) - 1
        centers[:,1] = centers[:,1] / (width/2) - 1
        centers = centers.reshape(1,1,centers.size(0), 2)
        tags = nn.functional.grid_sample(tags, centers).reshape([self.tag_dim, -1]).permute(1,0)
        
        if cfg.nms_type == 'soft_tag_nms':
            proposals, nms_idx = soft_tag_nms(proposals, tags, cfg.tag_thr, cfg.min_score)
        elif cfg.nms_type == 'tag_nms':
            proposals, nms_idx = tag_nms(proposals, tags, cfg.tag_thr)
        elif cfg.nms_type == 'nms':
            proposals, nms_idx = nms(proposals, cfg.nms_thr)
        rpn_cls_score = rpn_cls_score[nms_idx, :]
        proposals = proposals[:cfg.nms_post, :]
        rpn_cls_score = rpn_cls_score[:cfg.nms_post, :]
        

        scores = proposals[:, 4]
        num = min(cfg.max_num, proposals.shape[0])
        _, topk_inds = scores.topk(num)
        proposals = proposals[topk_inds, :]
        rpn_cls_score = rpn_cls_score[topk_inds, :]
        tags = tags[topk_inds, :]
        self.rpn_score = rpn_cls_score
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
