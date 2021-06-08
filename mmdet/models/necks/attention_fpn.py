import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from mmcv.cnn import xavier_init

from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class AttentionFPN(nn.Module):

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
                 min_act_height = 5,
                 act_feat_num = 1):
        super(AttentionFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.min_act_height = min_act_height
        self.act_feat_num = act_feat_num

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
        for i in range(self.act_feat_num):
            atten_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            atten_pred = nn.Conv2d(out_channels, 1, 1)
            
            self.atten_convs.append(atten_conv)
            self.atten_preds.append(atten_pred)


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
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        act_outs, atten_heatmaps = self.activate_feat(outs)
        self.atten_heatmaps = tuple(atten_heatmaps)
        return tuple(act_outs)

    def activate_feat(self, fpn_outs):
        assert len(fpn_outs) >= self.act_feat_num
        act_outs = []
        atten_heatmaps = []
        for i in range(self.act_feat_num):
            atten_feat = self.atten_convs[i](fpn_outs[i])
            atten_heatmap = self.atten_preds[i](atten_feat)
            atten_heatmaps.append(atten_heatmap)
            act_outs.append(fpn_outs[i])
            # act_feat = fpn_outs[i] * atten_heatmap.expand(-1, fpn_outs[i].size(1), -1, -1)
            # act_outs.append(act_feat)
        # append extra outs
        for i in range(self.act_feat_num, len(fpn_outs)):
            act_outs.append(fpn_outs[i])
        return act_outs, atten_heatmaps

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
        gauss_t = self.gen_gauss_template()
        feat_shapes = [ list(atten_heatmap.size()[2:]) for atten_heatmap in self.atten_heatmaps ][:self.act_feat_num]
        feat_shapes = feat_shapes[::-1] # from small to big
        feat_shapes = np.array(feat_shapes, dtype = np.float32)
        img_shapes = [ img_meta['pad_shape'][:2] for img_meta in img_metas ]
        all_gt_hmp = list()
        all_mask = list()
        for i in range(len(gt_bboxes)): # process single img
            gt_hmp, mask = self.gen_single_img_gt(gt_bboxes[i].cpu().numpy(),
                                                  gt_bboxes_ignore[i].cpu().numpy(),
                                                  img_shapes[i],
                                                  feat_shapes,
                                                  gauss_t)
            all_gt_hmp.append(gt_hmp)
            all_mask.append(mask)
        # just for debug
        #############
        self.gt_hmp_tensor = []

        ##############
        losses = 0
        for i in range(self.act_feat_num):
            cur_gt_hmp = np.array([ gt_hmp[i] for gt_hmp in all_gt_hmp ])[:,np.newaxis,:,:]
            cur_mask = np.array([ mask[self.act_feat_num - 1 - i] for mask in all_mask ])[:,np.newaxis,:,:]
            cur_pred = self.atten_heatmaps[i]
            cur_gt_hmp = cur_pred.new_tensor(cur_gt_hmp)
            self.gt_hmp_tensor.append(cur_gt_hmp)
            cur_mask = cur_pred.new_tensor(cur_mask)
            loss = (((cur_pred - cur_gt_hmp)**2) * cur_mask).mean()
            losses += loss
        losses /= self.act_feat_num
        return dict(loss_hmp = losses)
        
    def gen_single_img_gt(self, bboxes, ignore_bboxes, img_shape, feat_shapes, gauss_t, body_to_head = 6.5):
        # bboxes: [[x1, y1, x2, y2],...]
        # ignore_bboxes: [[x1, y1, x2, y2],...]
        # img_shape: [img_height, img_width]
        # feat_shapes:[[feat1_height, feat1_width],[feat2_height, feat2_width],...]
        #               from small to big
        # gauss_t: generated from gen_gauss_template()
        gt_heatmaps = list()
        gt_masks = list()
        b_heights = bboxes[:, 3] - bboxes[:, 1]
        sort_idx = np.argsort(b_heights)[::-1]
        sort_bboxes = bboxes[sort_idx]
        sort_bboxes = np.array(sort_bboxes, dtype = np.int) # quantify
        ori_hmp = np.zeros(img_shape, dtype = np.float32)
        ig_bboxes = np.array(ignore_bboxes, dtype = np.int) # quantify
        ori_mask = np.ones(img_shape, dtype = np.float32)
        strides = img_shape[0] / feat_shapes[:,0]
        height_thr = strides * self.min_act_height
        level = len(strides) - 1

        # generate gt_heatmaps for each level
        for bbox in sort_bboxes: # from big to small
            
            b_h = bbox[3] - bbox[1]
            b_w = bbox[2] - bbox[0]
            if b_h < height_thr[level]:
                cur_hmp = cv2.resize(ori_hmp, tuple(feat_shapes[level][::-1])) # (width, height)
                gt_heatmaps.append(cur_hmp)
                level -= 1
            if level < 0:
                break
            # g_box = cv2.resize(gauss_t, (b_w, b_h))
            # c,d = max(0, -bbox[0]), min(bbox[2], img_shape[1]) - bbox[0]
            # a,b = max(0, -bbox[1]), min(bbox[3], img_shape[0]) - bbox[1]
            # cc,dd = max(0, bbox[0]), min(bbox[2], img_shape[1])
            # aa,bb = max(0, bbox[1]), min(bbox[3], img_shape[0])
            # head_y = int(bbox[1] - (b_w - b_h/body_to_head)/2)
            # head_box = [bbox[0], head_y, bbox[2], head_y + b_w]
            head_box = [int(bbox[0] - b_w / 2),
                        int(bbox[1] - (b_w - b_h / 13)), 
                        int(bbox[2] + b_w / 2), 
                        int(bbox[1] + (b_w + b_h / 13))]
            g_box = cv2.resize(gauss_t, (2*b_w, 2*b_w))
            c,d = max(0, -head_box[0]), min(head_box[2], img_shape[1]) - head_box[0]
            a,b = max(0, -head_box[1]), min(head_box[3], img_shape[0]) - head_box[1]
            cc,dd = max(0, head_box[0]), min(head_box[2], img_shape[1])
            aa,bb = max(0, head_box[1]), min(head_box[3], img_shape[0])
            ori_hmp[aa:bb,cc:dd] = np.maximum(ori_hmp[aa:bb,cc:dd], g_box[a:b,c:d])
        for i in range(level, -1, -1):
            cur_hmp = cv2.resize(ori_hmp, tuple(feat_shapes[i][::-1]))
            gt_heatmaps.append(cur_hmp)
        
        # generate mask on ignore boxes:
        for bbox in ig_bboxes:
            cc,dd = max(0, bbox[0]), min(bbox[2], img_shape[1])
            aa,bb = max(0, bbox[1]), min(bbox[3], img_shape[0])
            ori_mask[aa:bb,cc:dd] = 0
        # generate masks for each level
        sort_bboxes = sort_bboxes[::-1]
        level = 0
        for bbox in sort_bboxes: # from small to big
            b_h = bbox[3] - bbox[1]
            b_w = bbox[2] - bbox[0]
            if b_h >= height_thr[level]:
                cur_mask = cv2.resize(ori_mask, tuple(feat_shapes[level][::-1])) # (width, height)
                gt_masks.append(cur_mask)
                level += 1
            if level >= len(strides):
                break
            cc,dd = max(0, bbox[0]), min(bbox[2], img_shape[1])
            aa,bb = max(0, bbox[1]), min(bbox[3], img_shape[0])
            ori_mask[aa:bb,cc:dd] = 0
        for i in range(level, len(strides)):
            cur_mask = cv2.resize(ori_mask, tuple(feat_shapes[i][::-1]))
            gt_masks.append(cur_mask)
        return gt_heatmaps, gt_masks

    def gen_gauss_template(self, sigma = 2):
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g
        
