from __future__ import division

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmdet import ops
from ..registry import ROI_EXTRACTORS
import copy

@ROI_EXTRACTORS.register_module
class Img_ROI_Extractor(nn.Module):
    """Extract RoI features from img.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
    """

    def __init__(self,
                 roi_layer,
                 expand_ratio):
        super(Img_ROI_Extractor, self).__init__()
        self.img_roi_layer = self.build_img_roi_layers(roi_layer)
        self.expand_ratio = expand_ratio

    def build_img_roi_layers(self, layer_cfg):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layer = layer_cls(spatial_scale=1, **cfg)
        return roi_layer

    def forward(self, imgs, rois):
        # rois: Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
        ori_rois = copy.deepcopy(rois)
        img_height = imgs.size(2)
        img_width = imgs.size(3)
        exp_heights = (ori_rois[:,4] - ori_rois[:,2]) * self.expand_ratio 
        exp_widths = (ori_rois[:,3] - ori_rois[:,1]) * self.expand_ratio 
        # for here is a bug in pytorch, just use clamp to trans to tensor
        ori_rois[:,1] = torch.clamp(ori_rois[:,1] - exp_widths, -img_width, 2*img_width)
        ori_rois[:,2] = torch.clamp(ori_rois[:,2] - exp_heights, -img_height, 2*img_height)
        ori_rois[:,3] = torch.clamp(ori_rois[:,3] + exp_widths, -img_width, 2*img_width)
        ori_rois[:,4] = torch.clamp(ori_rois[:,4] + exp_heights, -img_height, 2*img_height)

        return self.img_roi_layer(imgs, ori_rois), ori_rois
