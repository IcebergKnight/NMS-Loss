import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
import cv2
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np

@DETECTORS.register_module
class RPN_RGBRCNN(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 rcnn_backbone=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 rcnn_pretrained=None,
                 fix_rpn = False,
                 fix_rcnn = False,
                 generate_xgb_data = False):
        super(RPN_RGBRCNN, self).__init__()
        self.rpn_part = []
        self.rcnn_part = []
        self.fix_rpn = fix_rpn
        self.fix_rcnn = fix_rcnn
        self.generate_xgb_data = generate_xgb_data
        self.backbone = builder.build_backbone(backbone)
        self.rpn_part.append(self.backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
            self.rpn_part.append(self.neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)
            self.rpn_part.append(self.shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)
            self.rpn_part.append(self.rpn_head)

        if rcnn_backbone is not None:
            self.rcnn_backbone = builder.build_backbone(rcnn_backbone)
            self.rcnn_part.append(self.rcnn_backbone)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
            if hasattr(self.bbox_head, 'gt_masks'):
                rcnn_mask_extractor = copy.deepcopy(bbox_roi_extractor)
                rcnn_mask_extractor['roi_layer']['out_size'] = \
                                bbox_head['max_feat_size']
                self.rcnn_mask_extractor = builder.build_roi_extractor(
                    rcnn_mask_extractor)
            self.rcnn_part.append(self.bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained, rcnn_pretrained = rcnn_pretrained)
        self._freeze()

    def _freeze(self):
        if self.fix_rpn:
            for m in self.rpn_part:
                for param in m.parameters():
                    param.requires_grad = False

        if self.fix_rcnn:
            for m in self.rcnn_part:
                for param in m.parameters():
                    param.requires_grad = False
    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_rcnn_backbone(self):
        return hasattr(self, 'rcnn_backbone') and self.rcnn_backbone is not None

    def init_weights(self, pretrained=None, rcnn_pretrained=None):
        super(RPN_RGBRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_rcnn_backbone:
            self.rcnn_backbone.init_weights(rcnn_pretrained)
        if self.with_bbox:
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck: 
            x = self.neck(x)
        return x
    
    # just used for debug
    def visible_mask(self, img_tensor, mask_tensors, post_fix = 'pred'):
        alpha = 0.5
        dst_dir = '/home/joeyfang/fzcode/research/mmdetection/visible_mask_' + post_fix
        current_milli_time = lambda: int(round(time.time() * 1000))
        assert img_tensor.size()[0] == mask_tensors[0].size(0)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        # 1.get single img to numpy
        # 2.trans [c,h,w] to [h,w,c]
        # 3.trans rgb to bgr for using cv2
        # 4.get singlemask to numpy
        # 5.expand mask to img size
        # 6.merge img and mask
        # 7.save
        for i in range(min(img_tensor.size()[0], 30)):
            img = img_tensor[i].cpu().numpy()
            img = img.transpose(1,2,0)
            img = img * std + mean
            img = img[:,:,::-1]
            img_name = str(current_milli_time()) + '.jpg'
            sub_dir = os.path.join(dst_dir, 'ori')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            cv2.imwrite(os.path.join(sub_dir, img_name),img)
            for scale_idx, mask_tensor in enumerate(mask_tensors):
                mask = mask_tensor[i].detach().cpu().numpy()
                mask = mask.transpose(1,2,0).squeeze()
                print(mask.shape)
                print(img.shape)
                mask = cv2.resize(mask, img.shape[:2][::-1])
                # s_img = cv2.resize(img, mask.shape[::-1]).astype(np.int32)
                cmap = plt.get_cmap('jet')
                rgba_img = cmap(mask)
                rgb_img = np.delete(rgba_img, 3, 2)
                mask = ((rgb_img * 254)[:,:,::-1]).astype(np.int32)
                s_img = img.copy().astype(np.int32)
                cv2.addWeighted(mask, alpha, s_img, 1-alpha, 0, s_img)
                sub_dir = os.path.join(dst_dir, str(scale_idx))
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                merge_path = os.path.join(sub_dir, img_name)
                cv2.imwrite(merge_path,s_img)

    def visible_heatmap(self, img_tensor, mask_tensors, post_fix = 'pred'):
        alpha = 0.5
        dst_dir = '/home/joeyfang/fzcode/research/mmdetection/visible_heatmap_' + post_fix
        current_milli_time = lambda: int(round(time.time() * 1000))
        assert img_tensor.size()[0] == 1
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        # 1.get single img to numpy
        # 2.trans [c,h,w] to [h,w,c]
        # 3.trans rgb to bgr for using cv2
        # 4.get singlemask to numpy
        # 5.expand mask to img size
        # 6.merge img and mask
        # 7.save
        mask_tensors = mask_tensors[0][0]
        for i in range(img_tensor.size()[0]):
            img = img_tensor[i].cpu().numpy()
            img = img.transpose(1,2,0)
            img = img * std + mean
            img = img[:,:,::-1]
            img_name = str(current_milli_time()) + '.jpg'
            sub_dir = os.path.join(dst_dir, 'ori')
            # if not os.path.exists(sub_dir):
            #     os.makedirs(sub_dir)
            # cv2.imwrite(os.path.join(sub_dir, img_name),img)
            print(mask_tensors.size())
            for scale_idx, mask_tensor in enumerate(mask_tensors):
                mask = mask_tensor.detach().cpu().numpy()
                # only used for visible tag
                ###########################
                # med = np.median(mask)
                # print(med)
                # print('max:',np.max(mask), 'min:', np.min(mask))
                # bkg_idx = np.abs(mask-med) < 1
                # fg_idx = np.abs(mask-med) >= 1
                # mean = np.mean(mask[fg_idx])
                # print(mean)
                # mask[bkg_idx] = mean
                # mask -= np.min(mask)
                ###########################

                # mask = mask.transpose(1,2,0).squeeze()
                print(mask.shape)
                print(img.shape)
                mask = cv2.resize(mask, img.shape[:2][::-1])
                # s_img = cv2.resize(img, mask.shape[::-1]).astype(np.int32)
                cmap = plt.get_cmap('jet')
                rgba_img = cmap(mask)
                rgb_img = np.delete(rgba_img, 3, 2)
                mask = ((rgb_img * 254)[:,:,::-1]).astype(np.int32)
                s_img = img.copy().astype(np.int32)
                cv2.addWeighted(mask, alpha, s_img, 1-alpha, 0, s_img)
                sub_dir = os.path.join(dst_dir, str(scale_idx))
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                merge_path = os.path.join(sub_dir, img_name)
                cv2.imwrite(merge_path,s_img)
    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        losses = dict()

        if not self.fix_rpn or proposals is None:
            x = self.extract_feat(img)
            # some special FPN loss
            if hasattr(self.neck, 'loss_mask_weight') and not self.fix_rpn:
                neck_losses = self.neck.loss(gt_bboxes, img_meta, gt_bboxes_ignore)
                losses.update(neck_losses)
                # self.visible_mask(img, self.neck.gt_masks_tensor, 'gt')
                # self.visible_mask(img, self.neck.atten_masks, 'pred')

            # RPN forward and loss
            if self.with_rpn:
                rpn_outs = self.rpn_head(x)
                # self.visible_heatmap(img, rpn_outs[2], 'train_tag')
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                requires_grad = True # Pay attention here! 
                # if the rpn is nms_loss_rpn, it means 'requires_grad';
                # otherwise, means 'rescale'
                proposal_inputs = rpn_outs + (img_meta, proposal_cfg, requires_grad)
                proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
                
                rpn_loss_inputs = rpn_outs + (gt_bboxes, proposal_list, img_meta,
                                              self.train_cfg.rpn)
                rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
                losses.update(rpn_losses)

        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if (self.with_bbox or self.with_mask) and not self.fix_rcnn:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                # print(assign_result.max_overlaps)
                # print('assign gt_inds len')
                # print(len(assign_result.gt_inds))
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels=gt_labels[i],
                    feats=None)
                    # feats=[lvl_feat[i][None] for lvl_feat in x])
                # print('sampling pos_inds len')
                # print(len(sampling_result.pos_inds))
                # print('sampling neg_inds len')
                # print(len(sampling_result.neg_inds))
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox and not self.fix_rcnn:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # img_extrator to crop img
            if rois.size(0) == 0:
                losses.update({'loss_cls':rois.new_zeros((1), requires_grad = True),
                               'loss_rcnn_mask':rois.new_zeros((1), requires_grad=True)})
                return losses
            bbox_imgs, expand_rois = self.bbox_roi_extractor(img, rois)
            rcnn_feat = bbox_imgs
            if self.with_rcnn_backbone:
                rcnn_feat = self.rcnn_backbone(bbox_imgs)
            cls_score = self.bbox_head(rcnn_feat)
            if hasattr(self.bbox_head, 'gt_masks'):
                all_gt_masks, all_ig_masks = self.neck.gen_oriimg_size_masks(
                                        gt_bboxes, img_meta, gt_bboxes_ignore)
                # self.visible_mask(img, [all_gt_masks], 'gt')
                rcnn_gt_masks, expand_rois = self.rcnn_mask_extractor(all_gt_masks, rois)
                rcnn_ig_masks, expand_rois = self.rcnn_mask_extractor(all_ig_masks, rois)
                zero_tensor = torch.zeros_like(rcnn_gt_masks)
                one_tensor = torch.ones_like(rcnn_gt_masks)
                rcnn_gt_masks = torch.where(rcnn_gt_masks > 0.9, one_tensor, zero_tensor)
                rcnn_ig_masks = torch.where(rcnn_ig_masks < 0.1, zero_tensor, one_tensor)
                self.bbox_head.gt_masks = rcnn_gt_masks
                self.bbox_head.ig_masks = rcnn_ig_masks
            if hasattr(self.bbox_head, 'expand_rois'):
                self.bbox_head.expand_rois = expand_rois
            bbox_targets, pos_assigned_gt_inds = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_targets[0], bbox_targets[1])
            # self.visible_mask(bbox_imgs, [self.bbox_head.gt_masks, self.bbox_head.ig_masks], 'gt')
            losses.update(loss_bbox)

        # mask head forward and loss
        # if self.with_mask:
        #     if not self.share_roi_extractor:
        #         pos_rois = bbox2roi(
        #             [res.pos_bboxes for res in sampling_results])
        #         mask_feats = self.mask_roi_extractor(
        #             x[:self.mask_roi_extractor.num_inputs], pos_rois)
        #         if self.with_shared_head:
        #             mask_feats = self.shared_head(mask_feats)
        #     else:
        #         pos_inds = []
        #         device = bbox_feats.device
        #         for res in sampling_results:
        #             pos_inds.append(
        #                 torch.ones(
        #                     res.pos_bboxes.shape[0],
        #                     device=device,
        #                     dtype=torch.uint8))
        #             pos_inds.append(
        #                 torch.zeros(
        #                     res.neg_bboxes.shape[0],
        #                     device=device,
        #                     dtype=torch.uint8))
        #         pos_inds = torch.cat(pos_inds)
        #         mask_feats = bbox_feats[pos_inds]
        #     mask_pred = self.mask_head(mask_feats)

        #     mask_targets = self.mask_head.get_target(
        #         sampling_results, gt_masks, self.train_cfg.rcnn)
        #     pos_labels = torch.cat(
        #         [res.pos_gt_labels for res in sampling_results])
        #     loss_mask = self.mask_head.loss(mask_pred, mask_targets,
        #                                     pos_labels)
        #     losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)
        # self.visible_mask(img, self.neck.atten_masks, 'pred')
        # rpn_outs = self.rpn_head(x)[2]
        # self.visible_heatmap(img, rpn_outs, 'test_tag')

        ### retina mode
        # proposal_list, rpn_score_list, rpn_tag_list = self.rgb_simple_test_rpn(
        #     x, img_meta, self.test_cfg.rpn, self.neck.atten_masks) if proposals is None else (proposals, None, None)
        ###tmp, retina, !!!
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        rpn_score_list = self.rpn_head.get_rpn_scores()
        rpn_tag_list = self.rpn_head.get_rpn_tags()
        ###tmp, retina, !!!

        if self.fix_rcnn:
            if rpn_score_list is None or proposal_list[0] is None:
                return None
            rpn_score = rpn_score_list[0] # rpn score:size [sample_num, 2 or 1]
            if rpn_score.size(1) == 2: # use softmax
                scores = F.softmax(rpn_score, dim=1)
            else: # use sigmoid
                rpn_score = rpn_score.sigmoid()
                scores= torch.cat((1-rpn_score, rpn_score), 1)
            rois = bbox2roi(proposal_list)
            _bboxes = rois[:, 1:]
            scale_factor = img_meta[0]['scale_factor']
            if rescale:
                _bboxes /= scale_factor
            _scores = scores[:, 1]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
            cls_labels = _bboxes.new_full(
                        (cls_dets.shape[0], ), 0, dtype=torch.long)
            bbox_results = bbox2result(cls_dets, cls_labels, self.bbox_head.num_classes)
            #!!!! ONLY for visible TAG !!!!
            # rpn_tag = rpn_tag_list[0].cpu().numpy()
            # bbox_results = np.concatenate(bbox_results, 0)
            # bbox_results = np.concatenate((bbox_results, rpn_tag),-1)
            ###############################
            return bbox_results

        # print('proposal_list')
        # print(proposal_list)
        det_bboxes, det_labels, rcnn_score = self.rgb_simple_test_bboxes(
            x, img, img_meta, proposal_list, rpn_score_list,
            rpn_tag_list, self.test_cfg.rcnn, rescale=rescale)
        if det_bboxes is None:
            return None
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        #!!!! ONLY for visible TAG !!!!
        # rpn_tag = rpn_tag_list[0].cpu().numpy()
        # bbox_results = np.concatenate(bbox_results, 0)
        ###############################

        # print('bbox_results')
        # print(bbox_results)
        if self.generate_xgb_data:
            rpn_score = rpn_score_list[0]
            ped_res = bbox_results[0]
            ped_res = np.concatenate((ped_res[:,:4], 
                                      rpn_score.cpu().numpy(),
                                      rcnn_score.cpu().numpy()), axis = 1)
            bbox_results[0] = ped_res
            return bbox_results
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
