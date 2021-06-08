import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import delta2bbox, multiclass_nms, bbox_target
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from ..utils import ConvModule
from mmcv.cnn import xavier_init
import numpy as np
import cv2
import xgboost as xgb
import copy

@HEADS.register_module
class WekSegBBoxHead(nn.Module):
    # for new, we only use the c4 and c4 for prediction mask
    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=False,
                 max_feat_size=(14, 6),
                 num_classes=2,
                 in_channels = [1024, 2048],
                 out_channel = 256,
                 mask_feat_channel = 256,
                 global_fc_channel = 4096,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 loss_mask_weight=1,
                 dropout_rate = -1,
                 xgb_model = None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(WekSegBBoxHead, self).__init__()
        assert with_cls and len(in_channels) == 2
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.max_feat_size = max_feat_size
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.expand_rois = None
        self.gt_masks = None
        self.ig_masks = None
        self.loss_mask_weight = loss_mask_weight
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.cls_out_channels = num_classes - 1 if self.use_sigmoid_cls else num_classes
        self.xgb_merge = None
        if xgb_model is not None:
            self.xgb_merge = xgb.Booster({'nthread': 4})
            self.xgb_merge.load_model(xgb_model)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.in_channels = in_channels
        self.mask_feat_channel = mask_feat_channel
        self.out_channel = out_channel
        self.up_strides = np.array(in_channels) / in_channels[0]
        
        self.atten_convs = nn.ModuleList()
        self.atten_preds = nn.ModuleList()
        for in_channel in self.in_channels:
            atten_conv = ConvModule(
                in_channel,
                self.mask_feat_channel,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                activation=None,
                inplace=False)
            atten_pred = nn.Conv2d(self.mask_feat_channel, 1, 1)
            self.atten_convs.append(atten_conv)
            self.atten_preds.append(atten_pred)
        self.down_sample = nn.MaxPool2d(kernel_size=int(self.up_strides[-1]))
        self.atten_smooth = ConvModule(
            in_channels[-1] + self.mask_feat_channel * len(in_channels),
            out_channel,
            1,
            conv_cfg=None,
            norm_cfg=None,
            activation=None,
            inplace=False)
        self.final_feat_size = np.array(max_feat_size) / self.up_strides[-1]
        self.fc_in_channel = self.final_feat_size[0] *  self.final_feat_size[1] * out_channel
        self.fc_in_channel = int(self.fc_in_channel)
        self.fc_global1 = nn.Linear(self.fc_in_channel, global_fc_channel)
        self.fc_global2 = nn.Linear(global_fc_channel, global_fc_channel)
        if self.with_cls:
            self.fc_class = nn.Linear(global_fc_channel, self.cls_out_channels)
        if self.with_reg:
            self.fc_reg = nn.Linear(global_fc_channel, 4)

        self.drop1 = None
        if dropout_rate != -1:
            self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        nn.init.normal_(self.fc_global1.weight, 0, 0.01)
        nn.init.constant_(self.fc_global1.bias, 0)
        nn.init.normal_(self.fc_global2.weight, 0, 0.01)
        nn.init.constant_(self.fc_global2.bias, 0)
        if self.with_cls:
            nn.init.normal_(self.fc_class.weight, 0, 0.01)
            nn.init.constant_(self.fc_class.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        up_feats = []
        up_feats.append(inputs[0])
        for i in range(1, len(inputs)):
            up_feats.append(F.interpolate(
                    inputs[i], scale_factor=self.up_strides[i], mode='nearest'))
        masks = []
        mask_feats = []
        for idx, up_feat in enumerate(up_feats):
            mask_feat = self.atten_convs[idx](up_feat)
            mask = self.atten_preds[idx](mask_feat)
            mask_feats.append(mask_feat)
            masks.append(mask)
        mask_feats = torch.cat(mask_feats, dim = 1)
        mask_feats = self.down_sample(mask_feats)
        cat_feats = torch.cat((inputs[-1], mask_feats), dim = 1)
        smooth_feats = self.atten_smooth(cat_feats)
        x = smooth_feats.view(smooth_feats.size(0), -1)
        x = self.fc_global1(x)
        x = self.relu(x)
        x = self.fc_global2(x)
        x = self.relu(x)
        if self.drop1 is not None:
            x = self.drop1(x)
        cls_score = self.fc_class(x) if self.with_cls else None
        self.masks = masks
        # for now, not support offset regression 
        # bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        neg_overlaps = [res.max_overlaps[res.neg_inds] for res in sampling_results]
        reg_classes = self.num_classes
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        # print(pos_gt_bboxes[0].size())
        # print(pos_assigned_gt_inds[0].size())
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            pos_assigned_gt_inds = pos_assigned_gt_inds,
            neg_overlaps = neg_overlaps
            )
        labels = cls_reg_targets[0]
        # print(labels.size())
        # print(labels)
        label_weights = cls_reg_targets[1]

        # append height sensitive weight:
        # 1.caculate mean height: mean_height
        # 2. new_weight += cur_height / mean_height
        all_height = self.expand_rois[:, 4] - self.expand_rois[:, 2]
        mean_height = all_height.mean()
        label_weights += all_height / mean_height
        
        # WARNING: this method is DISCARDED right now.
        # generate mask gt
        # gt_bboxes = pos_gt_bboxes[0]
        # gt_masks = []
        # gt_idx = 0
        # for idx, label in enumerate(labels):
        #     mask = np.zeros(self.max_feat_size, dtype = np.int)
        #     if label != 0:
        #     #     mask = np.ones(self.max_feat_size, dtype = np.int)
        #     # if 1 != 1:
        #         gt_bbox = gt_bboxes[gt_idx] # [x1, y1, x2, y2]
        #         gt_idx += 1
        #         cur_roi = self.expand_rois[idx] # [batch_id, x1, y1, x2, y2]
        #         cur_roi = cur_roi[1:]
        #         gt_h = gt_bbox[3] - gt_bbox[1]
        #         gt_w = gt_bbox[2] - gt_bbox[0]
        #         cur_h = cur_roi[3] - cur_roi[1]
        #         cur_w = cur_roi[2] - cur_roi[0]
        #         new_gt_h = gt_h / (cur_h / self.max_feat_size[0])
        #         new_gt_w = gt_w / (cur_w / self.max_feat_size[1])
        #         rela_x1 = gt_bbox[0] - cur_roi[0]
        #         rela_y1 = gt_bbox[1] - cur_roi[1]
        #         rela_x1 = rela_x1 / (cur_w / self.max_feat_size[1])
        #         rela_y1 = rela_y1 / (cur_h / self.max_feat_size[0])
        #         new_gt_box = [rela_x1, rela_y1,
        #                       rela_x1 + new_gt_w, rela_y1 + new_gt_h]
        #         new_gt_box = np.array(new_gt_box).astype(int)
        #         cc = max(0, new_gt_box[0])
        #         dd = min(new_gt_box[2], self.max_feat_size[1])
        #         aa = max(0, new_gt_box[1])
        #         bb = min(new_gt_box[3], self.max_feat_size[0])
        #         mask[aa:bb,cc:dd] = 1
        #     gt_masks.append(mask)
        # self.gt_masks = np.array(gt_masks)[:,np.newaxis,:,:]
        # self.gt_masks = gt_bboxes.new_tensor(self.gt_masks)
        return (labels, label_weights), pos_assigned_gt_inds

    def loss(self,
             cls_score,
             labels,
             label_weights,
            reduction_override=None):
        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        # print(labels)
        losses['loss_cls'] = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=avg_factor,
            reduction_override=reduction_override)
        losses['acc'] = accuracy(cls_score, labels)
        mask_loss = 0
        for single_lev_mask in self.masks:
            s_mask_loss = F.binary_cross_entropy_with_logits(
                single_lev_mask, self.gt_masks, reduction='none')
            s_mask_loss *= self.ig_masks
            s_mask_loss = s_mask_loss.mean(dim = 3, keepdim=True) * label_weights
            s_mask_loss = s_mask_loss.mean()
            mask_loss += s_mask_loss
        losses['loss_rcnn_mask'] = (mask_loss * self.loss_mask_weight)
        return losses

    def get_det_bboxes(self, rois, cls_score, rpn_score_list, rpn_tag_list, img_shape, 
                        scale_factor, rescale=False, cfg=None):
        if isinstance(cls_score, list):
            print(cls_score)
            cls_score = sum(cls_score) / float(len(cls_score))
        rpn_score = rpn_score_list[0]  # rpn score:size [sample_num, 2 or 1]
        merge_mode = 'merge'
        if hasattr(cfg, 'merge_mode'):
            assert cfg.merge_mode in ['rpn','rcnn','merge']
            merge_mode = cfg.merge_mode

        if merge_mode == 'rpn':
            if rpn_score is None:
                scores = None
            elif rpn_score.size(1) == 2: # use softmax
                scores = F.softmax(rpn_score, dim=1)
            else: # use sigmod
                rpn_score = torch.squeeze(rpn_score, 1)
                rpn_score = rpn_score.sigmoid()
                scores = torch.cat((1-rpn_score, rpn_score), 1)
        elif merge_mode == 'rcnn':
            if cls_score is None:
                scores = None
            elif cls_score.size(1) == 2: # use softmax
                scores = F.softmax(cls_score, dim=1)
            else: # use sigmod
                cls_score = torch.squeeze(cls_score, 1)
                cls_score = cls_score.sigmoid()
                scores = torch.cat((1-cls_score, cls_score), 1)
        else:
            if rpn_score is None or cls_score is None:
                scores = None
            else:
                assert rpn_score.size(1) == cls_score.size(1)
                if self.xgb_merge is not None:
                    score_feat = np.concatenate((rpn_score.cpu().numpy(),
                                                cls_score.cpu().numpy()),
                                                axis = 1)
                    score_feat = xgb.DMatrix(score_feat)
                    scores = self.xgb_merge.predict(score_feat)
                    scores = np.concatenate(((1-scores)[:,None], 
                                                scores[:,None]), axis = 1)
                    scores = torch.from_numpy(scores).to(rpn_score.device)
                else:
                    rcnn_score_decay = 1
                    if hasattr(cfg, 'rcnn_score_decay'):
                        rcnn_score_decay = cfg.rcnn_score_decay
                    cls_score /= rcnn_score_decay
                    merge_score = cls_score + rpn_score
                    if rpn_score.size(1) == 2: # use softmax
                        scores = F.softmax(merge_score, dim=1)
                    else: # use sigmoid
                        merge_score = torch.squeeze(merge_score, 1)
                        merge_score = merge_score.sigmoid()
                        scores = torch.cat((1-merge_score, merge_score), 1)
        bboxes = rois[:, 1:]
        if rescale:
            bboxes /= scale_factor
        if not hasattr(cfg, 'nms') or cfg.nms is None:
            if scores is None:
                return None, None
            _scores = scores[:, 1]
            if hasattr(cfg, 'rec_score_value') and cfg.rec_score_value:
                 cls_dets = torch.cat([bboxes, _scores[:, None], 
                    rpn_score, cls_score, merge_score], dim=1)
            else:
                cls_dets = torch.cat([bboxes, _scores[:, None]], dim=1)
            cls_labels = bboxes.new_full(
                    (cls_dets.shape[0], ), 0, dtype=torch.long)
            return cls_dets, cls_labels
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, rpn_tag_list,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels

    def normalize_score(self, scores):
        assert scores.size(1) == 2
        mean = torch.mean(torch.abs(scores))
        scores /= mean

