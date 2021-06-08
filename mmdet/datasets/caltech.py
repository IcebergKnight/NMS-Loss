import numpy as np
import scipy.io
from .custom import CustomDataset
import pdb
import pickle
import copy

class CaltechDataset(CustomDataset):

    #CLASSES = ('ignore_regions' , 'pedestrians' , 'riders' , 'sitting_persons' , 'other_persons' , 'group_of_people')

    CLASSES = ('pedestrians')
    def load_annotations(self, ann_file):

        self.cat_ids = [1]
        raw_ann = pickle.load(open(ann_file,'rb'))
        self.img_infos = raw_ann['img_infos']
        for i in range(len(self.img_infos)):
            if self.img_infos[i]['filename'][-3:] != 'jpg':
                self.img_infos[i]['filename'] += '.jpg'
        self.annos = raw_ann['annos']
        return self.img_infos
       

    def get_ann_info(self, idx):
        ann_info = copy.deepcopy(self.annos[idx])
        return self._parse_ann_info(ann_info, self.with_mask)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        
        for i, img_info in enumerate(self.img_infos):
            #if self.img_ids[i] not in self.ids_with_ann:
            #    continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        print ('reasonable images: ' , len(valid_inds))
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=False):
        """Parse bbox and mask annotation.

        Args:
            ann_info (dict[array]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        gt_bboxes = ann_info['bboxes']
        gt_bboxes_ignore = ann_info['bboxes_ignore']
        gt_labels = ann_info['labels']
       

        if len(gt_bboxes_ignore) > 0:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if len(gt_bboxes) > 0:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            # filter box height < thr
            bboxes_height = gt_bboxes[:,3] - gt_bboxes[:,1] + 1
            valid_inds = np.where(bboxes_height >= 30)[0]
            ignore_inds = np.where(bboxes_height < 30)[0]
            gt_bboxes_ignore = np.concatenate((gt_bboxes_ignore, gt_bboxes[ignore_inds]), axis = 0)
            gt_bboxes = gt_bboxes[valid_inds]
            gt_labels = gt_labels[valid_inds]
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        
        
        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        return ann
