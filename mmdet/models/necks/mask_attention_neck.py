import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class MaskAttentionNeck(nn.Module):

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 mask_feat_channels=256,
                 loss_mask_weight=1,
                 merge_mask_feat = True, # wether merge mask feature to original fpn feat
                 output_indices=None,
                 input_indices = [-1]):
        super(MaskAttentionNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.mask_feat_channels = mask_feat_channels
        self.num_ins = len(in_channels)
        self.num_outs = len(in_channels)
        self.activation = activation
        self.loss_mask_weight = loss_mask_weight
        self.merge_mask_feat = merge_mask_feat
        self.output_indices = output_indices
        self.input_indices = input_indices

        self.atten_convs = nn.ModuleList()
        self.atten_preds = nn.ModuleList()
        for i in range(self.num_ins):
            atten_conv = ConvModule(
                self.in_channels[i],
                mask_feat_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            atten_pred = nn.Conv2d(mask_feat_channels, 1, 1)
            
            self.atten_convs.append(atten_conv)
            self.atten_preds.append(atten_pred)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        outs = []
        masks = []
        inputs = [ inputs[i] for i in self.input_indices]
        for i in range(self.num_ins):
            out = self.atten_convs[i](inputs[i])
            mask = self.atten_preds[i](out)
            out  = torch.cat((inputs[i], out), dim = 1)
            outs.append(out)
            masks.append(mask)

        self.atten_masks = tuple(masks)
        if self.output_indices is not None:
            outs = [outs[i] for i in self.output_indices]
        return tuple(outs)

    def loss(self, 
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = []
        # print(len(self.atten_heatmaps)) # 4:different scale featuremap count
        # print(self.atten_heatmaps[0].size()) # torch [batch,channel,height,width]
        # print(len(gt_bboxes)) # batch
        # print(gt_bboxes[0].size()) # torch [n, 4]
        # print(len(gt_bboxes_ignore)) # batch
        # print(gt_bboxes_ignore[0].size()) # torch [n, 4]
        # print(len(img_metas)) # batch
        # print(img_metas[0]) # dict: for caltech {'ori_shape': (480, 640, 3), 'img_shape': (768, 1024, 3), 'pad_shape': (768, 1024, 3), 'scale_factor': array([2., 2., 2., 2.], dtype=float32), 'flip': False}
        feat_shapes = [ list(atten_mask.size()[2:]) for atten_mask in self.atten_masks ]
        feat_shapes = np.array(feat_shapes, dtype = np.float32)
        img_shapes = [ img_meta['pad_shape'][:2] for img_meta in img_metas ]
        all_gt_masks = list()
        all_ig_masks = list()
        for i in range(len(gt_bboxes)): # process single img
            gt_masks, ig_masks = self.gen_single_img_gt(gt_bboxes[i].cpu().numpy(),
                                                  gt_bboxes_ignore[i].cpu().numpy(),
                                                  img_shapes[i],feat_shapes)
            all_gt_masks.append(gt_masks)
            all_ig_masks.append(ig_masks)
        # just for debug
        #############
        self.gt_masks_tensor = []

        ##############
        losses = 0
        for i in range(self.num_ins):
            cur_gt_mask = np.array([ gt_mask[i] for gt_mask in all_gt_masks ])[:,np.newaxis,:,:]
            cur_ig_mask = np.array([ ig_mask[i] for ig_mask in all_ig_masks ])[:,np.newaxis,:,:]
            cur_pred = self.atten_masks[i]
            cur_gt_mask = cur_pred.new_tensor(cur_gt_mask)
            self.gt_masks_tensor.append(cur_gt_mask)
            cur_ig_mask = cur_pred.new_tensor(cur_ig_mask)
            loss = F.binary_cross_entropy_with_logits(
                cur_pred, cur_gt_mask, reduction='none')
            loss = (loss * cur_ig_mask).mean()
            losses += loss
        losses /= self.num_ins
        return dict(loss_mask = losses * self.loss_mask_weight)
        
    def gen_single_img_gt(self, bboxes, ignore_bboxes, img_shape, feat_shapes):
        # bboxes: [[x1, y1, x2, y2],...]
        # ignore_bboxes: [[x1, y1, x2, y2],...]
        # img_shape: [img_height, img_width]
        # feat_shapes:[[feat1_height, feat1_width],[feat2_height, feat2_width],...]
        #               from big to small
        # return: gt_masks, ig_masks   from big to small
        gt_masks = list()
        ig_masks = list()
        strides = img_shape[0] / feat_shapes[:,0]

        # generate gt_masks for each level
        for idx, stride in enumerate(strides):
            cur_shape = feat_shapes[idx].astype(np.int)
            mask = np.zeros(cur_shape, dtype = np.int)
            ig_mask = np.ones(cur_shape, dtype = np.int)
            for bbox in bboxes:
                cur_bbox = (bbox/stride).astype(np.int)
                if cur_bbox[3] - cur_bbox[1] <=0 or cur_bbox[2] - cur_bbox[0] <=0:
                    continue
                cc,dd = max(0, cur_bbox[0]), min(cur_bbox[2], cur_shape[1])
                aa,bb = max(0, cur_bbox[1]), min(cur_bbox[3], cur_shape[0])
                mask[aa:bb,cc:dd] = 1
            gt_masks.append(mask)

            for ig_bbox in ignore_bboxes:
                cur_bbox = (ig_bbox/stride).astype(np.int)
                if cur_bbox[3] - cur_bbox[1] <=0 or cur_bbox[2] - cur_bbox[0] <=0:
                    continue
                cc,dd = max(0, cur_bbox[0]), min(cur_bbox[2], cur_shape[1])
                aa,bb = max(0, cur_bbox[1]), min(cur_bbox[3], cur_shape[0])
                ig_mask[aa:bb,cc:dd] = 0
            ig_masks.append(ig_mask)
        return gt_masks, ig_masks
    
    # this method is used to generate image size mask gt, for rcnn seg to use
    def gen_oriimg_size_masks(self, 
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = []
        img_shapes = [ img_meta['pad_shape'][:2] for img_meta in img_metas ]
        all_gt_masks = list()
        all_ig_masks = list()
        for i in range(len(gt_bboxes)): # process single img
            gt_masks, ig_masks = self.gen_single_img_gt(gt_bboxes[i].cpu().numpy(),
                            gt_bboxes_ignore[i].cpu().numpy(), img_shapes[i],
                            np.array([[img_shapes[i][0], img_shapes[i][1]]]))
            all_gt_masks.append(gt_masks[0])
            all_ig_masks.append(ig_masks[0])
        all_gt_masks = np.array(all_gt_masks)[:,np.newaxis,:,:]
        all_ig_masks = np.array(all_ig_masks)[:,np.newaxis,:,:]
        all_gt_masks = gt_bboxes[0].new_tensor(all_gt_masks)
        all_ig_masks = gt_bboxes[0].new_tensor(all_ig_masks)
        # all_gt_masks = torch.tensor(all_gt_masks, device = gt_bboxes[0].device)
        # all_ig_masks = torch.tensor(all_ig_masks, device = gt_bboxes[0].device)

        return all_gt_masks, all_ig_masks
