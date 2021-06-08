import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..registry import LOSSES

def cross_entropy(pred, label, weight=None, reduction='mean', 
                    avg_factor=None, regulation_thr=-1, regulation_weight = 1e-4):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')
    if regulation_thr >= 0:
        pred_reg = torch.sum(pred*pred, dim=1)
        pred_reg -= regulation_thr
        pred_reg = torch.clamp(pred_reg, min=0)
        pred_reg *= regulation_weight
        loss += pred_reg

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def smooth_cross_entropy(pred, label, weight=None, 
            reduction='mean', avg_factor=None, eps = 0.1):
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, label.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    # print(one_hot)
    log_prb = F.log_softmax(pred, dim=1)
    # print(log_prb)
    # minus min loss: for two class that is 0.325
    MIN_LOSS = 0.325
    loss = -(one_hot * log_prb).sum(dim=1) - MIN_LOSS
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

    

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         regulation_thr=-1,
                         regulation_weight = 1e-4):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    if regulation_thr >= 0:
        print(pred.size())
        pred_reg = torch.sum(pred*pred, dim=1)
        pred_reg -= regulation_thr
        pred_reg = torch.clamp(pred_reg, min=0)
        pred_reg *= regulation_weight
        loss += pred_reg
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0,
                 use_smooth = False,
                 regulation_thr = -1,
                 regulation_weight = 1e-3):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.use_smooth = use_smooth
        self.regulation_thr = regulation_thr
        self.regulation_weight = regulation_weight
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        elif self.use_smooth:
            self.cls_criterion = smooth_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.regulation_thr>=0:
            kwargs['regulation_thr'] = self.regulation_thr
            kwargs['regulation_weight'] = self.regulation_weight
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
