import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from .anchor_head import AnchorHead
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
from ..builder import build_loss
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox,
                        multi_apply, multiclass_nms)

@HEADS.register_module
class RetinaHeadNMS(AnchorHead):

    def __init__(self,
                num_classes,
                in_channels,
                nms_loss,
                stacked_convs=4,
                octave_base_scale=4,
                scales_per_octave=3,
                conv_cfg=None,
                norm_cfg=None,
                **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaHeadNMS, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)
        self.nms_loss = build_loss(nms_loss)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
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

        all_loss = dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

        assert hasattr(self, 'rpn_anchor_list'), 'Call get_bboxes() first before loss()'
        return all_loss

        # if len(proposal_list) == 0:
        #     all_loss['nms_push_loss'] = 0 * cls_scores[0].mean()
        #     all_loss['nms_pull_loss'] = 0 * cls_scores[0].mean()
        #     return all_loss

        # imgs_gt_inds = []
        # imgs_anchor_gt_inds = []
        # img_num = bbox_preds[0].size()[0]
        # for img_idx in range(img_num):
        #     proposals = proposal_list[img_idx]
        #     proposals_anchor = self.rpn_anchor_list[img_idx]
        #     # assign gt for each proposals
        #     cur_gt_box = gt_bboxes[img_idx]
        #     cur_gt_box_ignore = gt_bboxes_ignore[img_idx] if gt_bboxes_ignore is not None else None
        #     assign_results = self.assigner.assign(proposals, cur_gt_box, cur_gt_box_ignore)
        #     imgs_gt_inds.append(assign_results.gt_inds - 1) # ori: -1:ignore; 0:neg; pos num: pos;   after -1: <0:ignore; >=0:pos
        #     anchor_assign_results = self.assigner.assign(proposals_anchor, cur_gt_box, cur_gt_box_ignore)
        #     imgs_anchor_gt_inds.append(anchor_assign_results.gt_inds - 1) # ori: -1:ignore; 0:neg; pos num: pos;   after -1: <0:ignore; >=0:pos
        # losses_nms = self.nms_loss(imgs_gt_inds, imgs_anchor_gt_inds, gt_bboxes, proposal_list)
        # if isinstance(losses_nms, dict):
        #     all_loss.update(losses_nms)
        # else:
        #     all_loss['nms_loss'] = losses_nms
        # return all_loss 
