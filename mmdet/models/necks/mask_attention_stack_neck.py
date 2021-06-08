import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class MaskAttentionStackNeck(nn.Module):

    def __init__(self,
                 in_channels,
                 up_strides, # up_sample featuremap height
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 mask_feat_channels=256,
                 deconv_feat_channel = 256,
                 loss_mask_weight=1,
                 merge_mask_feat = True, # wether merge mask feature to original fpn feat
                 input_indices = None):
        super(MaskAttentionStackNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.mask_feat_channels = mask_feat_channels
        self.deconv_feat_channel = deconv_feat_channel
        self.up_strides = np.array(up_strides)
        assert np.max(self.up_strides) <= 4
        self.padding = ((4 - self.up_strides) / 2).astype(np.int8)
        self.num_ins = len(in_channels)
        self.activation = activation
        self.loss_mask_weight = loss_mask_weight
        self.merge_mask_feat = merge_mask_feat
        self.input_indices = input_indices
        self.num_norm_layers = self.num_ins + merge_mask_feat

        self.atten_conv = ConvModule(
            self.in_channels[-1],
            mask_feat_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=self.activation,
            inplace=False)
        self.atten_pred = nn.Conv2d(mask_feat_channels, 1, 1)

        self.l2_norm_list = nn.ModuleList()
        for i in range(self.num_ins):            
            self.l2_norm_list.append(L2Norm(self.deconv_feat_channel))

        self.deconv_list = nn.ModuleList()
        for i in range(self.num_ins):
            if self.up_strides[i]>1:
                self.deconv_list.append(nn.ConvTranspose2d(
                    self.in_channels[i],
                    self.deconv_feat_channel,
                    4,
                    self.up_strides[i],
                    self.padding[i]))

        self.conv_list = nn.ModuleList()
        for i in range(self.num_ins):
            if self.up_strides[i]==1:
                self.conv_list.append(nn.Conv2d(
                    self.in_channels[i], self.deconv_feat_channel, 1))
                
        if self.merge_mask_feat:
            self.l2_norm_list.append(L2Norm(self.deconv_feat_channel))
            if self.up_strides[-1]>1:
                self.deconv_list.append(nn.ConvTranspose2d(
                    mask_feat_channels,
                    self.deconv_feat_channel,
                    4,
                    self.up_strides[-1],
                    self.padding[-1]))
            else:
                self.conv_list.append(nn.Conv2d(
                    mask_feat_channels, self.deconv_feat_channel, 1))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_init(m, distribution='uniform')
        for l2_norm in self.l2_norm_list:
            l2_norm.init_weight()

    def forward(self, inputs):
        inputs = list(inputs)
        if self.input_indices is not None:
            inputs = [ inputs[i] for i in self.input_indices]
        mask_feat = self.atten_conv(inputs[-1])
        masks = [self.atten_pred(mask_feat)]
        deconv_idx = 0
        conv_idx = 0
        for i in range(len(inputs)):
            if self.up_strides[i]>1:
                inputs[i] = self.deconv_list[deconv_idx](inputs[i])
                deconv_idx += 1
            else:
                inputs[i] = self.conv_list[conv_idx](inputs[i])
                conv_idx += 1
            inputs[i] = self.l2_norm_list[i](inputs[i])
        out = torch.cat(inputs, dim = 1)

        if self.merge_mask_feat:
            if self.up_strides[-1]>1:
                mask_feat = self.deconv_list[-1](mask_feat)
            else:
                mask_feat = self.conv_list[-1](mask_feat)
            mask_feat = self.l2_norm_list[-1](mask_feat)
            out = torch.cat([out, mask_feat], dim = 1)

        outs = [out]
        self.atten_masks = tuple(masks)
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
        mask_num = len(self.atten_masks)
        for i in range(mask_num):
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
        losses /= mask_num
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

class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=10., eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def init_weight(self):
        torch.nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
