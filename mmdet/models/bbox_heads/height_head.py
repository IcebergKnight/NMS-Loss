import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import delta2bbox_nowidth, multiclass_nms, bbox_target
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS


@HEADS.register_module
class HeightHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 with_tag=False,
                 height_width_ratio = 1/0.41,
                 tag_dim=2,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_tag = dict(type='TagLoss', loss_weight=0.001)):
        super(HeightHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.with_tag = with_tag
        self.height_width_ratio = height_width_ratio
        self.tag_dim = tag_dim
        if type(roi_feat_size) is not tuple:
            roi_feat_size = (roi_feat_size, roi_feat_size)
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_tag = build_loss(loss_tag)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size[0] * self.roi_feat_size[1])
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg: # only predict offset_x, offset_y, delta_height
            out_dim_reg = 3 if reg_class_agnostic else 3 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        if self.with_tag:
            self.fc_tag = nn.Linear(in_channels, self.tag_dim)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_tag:
            nn.init.normal_(self.fc_tag.weight, 0, 0.01)
            nn.init.constant_(self.fc_tag.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        tag_pred = self.fc_tag(x) if self.with_tag else None
        return cls_score, bbox_pred, tag_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        # pos_assigned_gt_inds = [res.pos_assigned_gt_inds + idx * 1000 for idx, res in enumerate(sampling_results)] # multipl 1000 is used for avoiding different img tag overlapping
        # print(pos_assigned_gt_inds)
        # pos_assigned_gt_inds = torch.cat(pos_assigned_gt_inds).view(-1)
        # print(pos_assigned_gt_inds)
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        return cls_reg_targets, pos_assigned_gt_inds

    def loss(self,
             cls_score,
             bbox_pred,
             tag_pred,
             pos_assigned_gt_inds,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            # print(torch.nonzero(pos_inds).size(0))
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 3)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               3)[pos_inds, labels[pos_inds]]
            no_width_bbox_targets = bbox_targets[:,[0,1,3]]
            no_width_bbox_weights = bbox_weights[:,[0,1,3]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                no_width_bbox_targets[pos_inds],
                no_width_bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        if tag_pred is not None:
            pos_inds = labels > 0
            pos_tag_pred = tag_pred.view(tag_pred.size(0), self.tag_dim)[pos_inds]
            losses['loss_tag'] = self.loss_tag(
                pos_tag_pred,
                pos_assigned_gt_inds,
                avg_factor=pos_tag_pred.size(0),
                reduction_override=reduction_override)
            
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       tag_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            # added by joeyfang: img_shape will clip the box, so for no clipping, set img_shape = None
            img_shape = None
            # add end
            bboxes = delta2bbox_nowidth(rois[:, 1:], bbox_pred, self.height_width_ratio, self.target_means, self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:]
            # TODO: add clip here

        if rescale:
            bboxes /= scale_factor

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, tag_pred,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
