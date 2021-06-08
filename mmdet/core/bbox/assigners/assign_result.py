import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None, anchors_index=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self.anchors_index = anchors_index

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
        if self.anchors_index is not None:
            self.anchors_index = torch.cat(
                [-self.anchors_index.new_ones(self.num_gts), self.anchors_index])
