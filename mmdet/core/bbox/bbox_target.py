import torch

from .transforms import bbox2delta
from ..utils import multi_apply


def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                pos_assigned_gt_inds = None,
                neg_overlaps = None,
                concat=True):
    
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        pos_assigned_gt_inds,
        neg_overlaps = neg_overlaps,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


# single image
def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       pos_assigned_gt_inds,
                       neg_overlaps,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        if hasattr(cfg,'adaptive_pos_weight') and cfg.adaptive_pos_weight: #using adaptive weight for positive
            target_pos_num = cfg['sampler'].num * cfg['sampler'].pos_fraction
            all_pos_num = num_pos
            cat_num_rec = {}
            for ind in pos_assigned_gt_inds:
                ind = int(ind.cpu().numpy())
                if ind not in cat_num_rec.keys():
                    cat_num_rec[ind] = 0
                cat_num_rec[ind] += 1
            cat_num = len(cat_num_rec.keys())
            # print(pos_assigned_gt_inds)
            # print(cat_num_rec)
            # print(pos_assigned_gt_inds[0])
            pos_weight = pos_bboxes.new_full(pos_assigned_gt_inds.size(), pos_weight)
            for idx, ind in enumerate(pos_assigned_gt_inds):
                ind = int(ind.cpu().numpy())
                pos_weight[idx] *= (target_pos_num / cat_num / cat_num_rec[ind])
            # print(pos_weight)
        label_weights[:num_pos] = pos_weight
        pos_bboxes = pos_bboxes[:,:4]
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
        if hasattr(cfg,'smooth_assign_loss_alpha') and \
            cfg.smooth_assign_loss_alpha >= 0:
            sm_as_alpha = cfg.smooth_assign_loss_alpha
            assert sm_as_alpha < cfg.assigner['neg_iou_thr']
            a = 1 / (cfg.assigner['neg_iou_thr'] - sm_as_alpha)**2
            select_idx = neg_overlaps >= sm_as_alpha
            neg_weights = label_weights[-num_neg:]
            neg_weights[select_idx] = 1-a*(neg_overlaps[select_idx] - sm_as_alpha)**2
            label_weights[-num_neg:] = neg_weights
            # print(label_weights[-num_neg:])

    return labels, label_weights, bbox_targets, bbox_weights


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros((bbox_targets.size(0),
                                                  4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros((bbox_weights.size(0),
                                                  4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
