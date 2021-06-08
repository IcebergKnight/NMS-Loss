import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class MaskAttentionFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 mask_feat_channels=256,
                 act_feat_num = 3,
                 loss_mask_weight=1,
                 merge_mask_feat = True, # wether merge mask feature to original fpn feat
                 merge_mask2all = False, # wether merge last mask to all scale fpn feat
                 output_indices=None):
        super(MaskAttentionFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_feat_channels = mask_feat_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.act_feat_num = act_feat_num
        self.loss_mask_weight = loss_mask_weight
        self.merge_mask_feat = merge_mask_feat
        self.merge_mask2all = merge_mask2all
        self.output_indices = output_indices

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        
        self.atten_convs = nn.ModuleList()
        self.atten_preds = nn.ModuleList()
        self.atten_smooths = nn.ModuleList()
        for i in range(self.act_feat_num):
            atten_conv = ConvModule(
                out_channels,
                mask_feat_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            atten_pred = nn.Conv2d(mask_feat_channels, 1, 1)
            atten_smooth = ConvModule(
                out_channels+mask_feat_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            
            self.atten_convs.append(atten_conv)
            self.atten_preds.append(atten_pred)
            self.atten_smooths.append(atten_smooth)

        if self.merge_mask2all:
            for i in range(self.act_feat_num, num_outs):
                atten_smooth = ConvModule(
                    out_channels+mask_feat_channels,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.atten_smooths.append(atten_smooth)
                


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        # firstly, build layers with attention_mask
        used_backbone_levels = len(laterals)
        bottom_up_mask_num = self.num_outs - used_backbone_levels
        top_down_mask_num = self.act_feat_num - bottom_up_mask_num

        top_down_masks=[]
        for i in range(used_backbone_levels - 1, -1, -1):
            att_idx = i - (used_backbone_levels - top_down_mask_num)
            if att_idx >= 0: 
                # 1.generate mask feature
                mask_feat = self.atten_convs[att_idx](laterals[i])
                # 2.predict mask and append for caculating loss
                top_down_masks.append(
                    self.atten_preds[att_idx](mask_feat))
                if self.merge_mask_feat:
                    # 3.concate mask_feat and lateral_feat
                    laterals[i] = torch.cat((laterals[i], mask_feat), dim = 1)
                    # 4.use 1x1 conv to change channel
                    laterals[i] = self.atten_smooths[att_idx](laterals[i])
            elif self.merge_mask2all:
                # upsample mask feature to target tensor size
                t_size = [laterals[i].size(2), laterals[i].size(3)]
                upsample = nn.Upsample(size = t_size, mode='bilinear', align_corners=True)
                up_mask_feat = upsample(mask_feat)
                # concate mask_feat and lateral_feat
                laterals[i] = torch.cat((laterals[i], up_mask_feat), dim = 1)
                # use 1x1 conv to change channel
                laterals[i] = self.atten_smooths[att_idx](laterals[i])


            if i-1>=0: # when i = 0, skip
                laterals[i - 1] += F.interpolate(
                    laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels, from big to small
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        bottom_up_masks=[]
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
                    att_idx = top_down_mask_num + i
                    mask_feat = self.atten_convs[att_idx](outs[-1])
                    bottom_up_masks.append(
                        self.atten_preds[att_idx](mask_feat))
                    if self.merge_mask_feat:
                        outs[-1] = torch.cat((outs[-1], mask_feat), dim = 1)
                        outs[-1] = self.atten_smooths[att_idx](outs[-1])
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))

                att_idx = top_down_mask_num 
                mask_feat = self.atten_convs[att_idx](outs[-1])
                bottom_up_masks.append(
                    self.atten_preds[att_idx](mask_feat))
                if self.merge_mask_feat:
                    outs[-1] = torch.cat((outs[-1], mask_feat), dim = 1)
                    outs[-1] = self.atten_smooths[att_idx](outs[-1])
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

                    att_idx = top_down_mask_num + (i - used_backbone_levels)
                    mask_feat = self.atten_convs[att_idx](outs[-1])
                    bottom_up_masks.append(
                        self.atten_preds[att_idx](mask_feat))
                    if self.merge_mask_feat:
                        outs[-1] = torch.cat((outs[-1], mask_feat), dim = 1)
                        outs[-1] = self.atten_smooths[att_idx](outs[-1])
        # all_mask results: from big to small
        self.atten_masks = top_down_masks[::-1] + bottom_up_masks
        self.atten_masks = tuple(self.atten_masks)
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
        for i in range(self.act_feat_num):
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
        losses /= self.act_feat_num
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
