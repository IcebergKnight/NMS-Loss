import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (VGG, xavier_init, constant_init, kaiming_init,
                      normal_init)
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES


@BACKBONES.register_module
class RCNNVGG(VGG):

    def __init__(self,
                 depth,
                 with_bn=False,
                 num_stages=5,
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(0, 1, 2, 3, 4)):
        super(RCNNVGG, self).__init__(
            depth,
            with_bn=with_bn,
            num_stages=num_stages,
            frozen_stages=frozen_stages,
            bn_eval=bn_eval,
            bn_frozen=bn_frozen,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices)
