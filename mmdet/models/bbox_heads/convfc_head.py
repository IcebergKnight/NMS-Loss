import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.core import multiclass_nms
from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule
from ..losses import accuracy
from mmcv.cnn import constant_init, kaiming_init

@HEADS.register_module
class ConvFCHead(BBoxHead):

    def __init__(self,
                 num_convs=0,
                 num_fcs=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCHead, self).__init__(
                with_cls=True,
                with_reg=False,
                *args, 
                **kwargs)
        assert (num_convs + num_fcs > 0) 
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.convs, self.fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_convs, self.num_fcs, self.in_channels,
                True)
        self.cls_last_dim = last_layer_dim

        if self.num_fcs == 0 and not self.with_avg_pool:
            self.cls_last_dim *= (self.roi_feat_size[0] * self.roi_feat_size[1])

        self.relu = nn.ReLU(inplace=True)
        self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_size[0] * self.roi_feat_size[1])
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCHead, self).init_weights()
        for module_list in [self.fcs, self.fc_cls, self.convs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m)

    def forward(self, x):
        x = x[0]
        # shared part
        if self.num_convs > 0:
            for conv in self.convs:
                x = conv(x)

        if self.num_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.fcs:
                x = self.relu(fc(x))
        x_cls = x

        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)

        cls_score = self.fc_cls(x_cls) 
        return cls_score

    def loss(self,
             cls_score,
             labels,
             label_weights,
            reduction_override=None):
        # print(label_weights)
        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        losses['loss_cls'] = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=avg_factor,
            reduction_override=reduction_override)
        losses['acc'] = accuracy(cls_score, labels)
        return losses

    def get_det_bboxes(self, rois, cls_score, rpn_score_list, img_shape, 
                        scale_factor, rescale=False, cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        rpn_score = rpn_score_list[0]  # rpn score:size [sample_num, 2]
        merge_mode = 'merge'
        if hasattr(cfg, 'merge_mode'):
            assert cfg.merge_mode in ['rpn','rcnn','merge']
            merge_mode = cfg.merge_mode

        if merge_mode == 'rpn':
            if rpn_score.size(1) == 2: # use softmax
                scores = F.softmax(rpn_score, dim=1) if rpn_score is not None else None
            else: # use sigmod
                scores = torch.cat((1-rpn_score, rpn_score), 1)
        elif merge_mode == 'rcnn':
            scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            if rpn_score.size(1) == 2: # rpn use softmax
                rpn_score = F.softmax(rpn_score, dim=1) if rpn_score is not None else None
                cls_score += rpn_score
                scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
            else: # rpn use sigmoid
                rpn_scores = torch.cat((1-rpn_score, rpn_score), 1)
                rcnn_scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
                scores = (rpn_scores + rcnn_scores) / 2
        bboxes = rois[:, 1:]
        if rescale:
            bboxes /= scale_factor
        if cfg is None:
            _scores = scores[:, 1]
            cls_dets = torch.cat([bboxes, _scores[:, None]], dim=1)
            cls_labels = _bboxes.new_full(
                    (cls_dets.shape[0], ), 0, dtype=torch.long)
            return cls_dets, cls_labels
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, None,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels

