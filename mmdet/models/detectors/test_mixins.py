from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms)


class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def rgb_simple_test_rpn(self, x, img_meta, rpn_test_cfg, neck_mask):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        if hasattr(rpn_test_cfg, 'use_mask') and rpn_test_cfg.use_mask and neck_mask is not None and hasattr(self.rpn_head, 'neck_mask'):
            self.rpn_head.neck_mask = neck_mask
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        rpn_score_list = self.rpn_head.get_rpn_scores()
        rpn_tag_list = self.rpn_head.get_rpn_tags()
        return proposal_list, rpn_score_list, rpn_tag_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, rpn_test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           img=None):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        # proposals shape: [image_num, proposal_num, box+score+tag]
        if hasattr(self.bbox_roi_extractor,'ori_img'):
            self.bbox_roi_extractor.set_ori_img(img)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        rpn_tag = None
        if len(proposals[0]) > 0 and  proposals[0].size()[1] > 5:
            rpn_tag = proposals[0][:,5:]
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred, tag_pred = self.bbox_head(roi_feats)
        if tag_pred is None:
            tag_pred = rpn_tag
        # tag shape: [proposal_num, tag_dim]
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            tag_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def rgb_simple_test_bboxes(self,
                           x,
                           img,
                           img_meta,
                           proposals,
                           rpn_score_list,
                           rpn_tag_list,
                           rcnn_test_cfg,
                           rescale=False,
                           extra_feats = None,
                           visi_fun = None):
        """Test only det bboxes without augmentation."""
        if len(proposals) == 0:
            return None,None,None
        rois = bbox2roi(proposals)
        if rois.size(0) == 0:
            return None,None,None
        # proposals shape: [image_num, proposal_num, box+score+tag]
        roi_imgs, expand_rois = self.bbox_roi_extractor(img, rois)
        
        if self.with_rcnn_backbone:
            if extra_feats is not None:
                roi_feats, _ = self.feat_roi_extractor(extra_feats, rois)
                rcnn_feat = self.rcnn_backbone(roi_imgs, merge_feat = roi_feats)
            else:
                rcnn_feat = self.rcnn_backbone(roi_imgs)
        cls_score = self.bbox_head(rcnn_feat)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        # visi_fun(roi_imgs, self.bbox_head.masks, 'pred')
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            rpn_score_list,
            rpn_tag_list,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels, cls_score

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
