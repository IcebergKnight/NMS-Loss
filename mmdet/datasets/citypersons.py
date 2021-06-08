import numpy as np
import scipy.io
from .custom import CustomDataset
import pdb

class CityPersonsDataset(CustomDataset):

    #CLASSES = ('ignore_regions' , 'pedestrians' , 'riders' , 'sitting_persons' , 'other_persons' , 'group_of_people')

    CLASSES = ('pedestrians')
    def load_annotations(self, ann_file):

        self.cat_ids = [1]      
        anno_mat = scipy.io.loadmat(ann_file)
        if 'train' in ann_file:
            anno = anno_mat['anno_train_aligned'][0]

        if 'val' in ann_file:
            anno = anno_mat['anno_val_aligned'][0]

        self.img_ids = list()
        self.annos = list()
        self.ids_with_ann = list()
        img_infos = list()

        ignore_bboxes = 0
        train_bboxes = 0
        all_bboxes = 0

        for i, each_anno in enumerate(anno):
            img_name = each_anno[0][0][1][0]
            city_name = each_anno[0][0][0][0]
            bbs = each_anno[0][0][2]
            
            info = dict()
            #info['city_name'] = city_name
            info['width'] = 2048
            info['height']= 1024
            info['filename'] = img_name
            
            ann_info = self._parse_ann_info(bbs)
            if len(ann_info['labels']) != 0:
                self.ids_with_ann.append(i+1)

            ignore_bboxes = ignore_bboxes + len(ann_info['bboxes_ignore'])
            train_bboxes = train_bboxes + len(ann_info['bboxes'])
            all_bboxes = all_bboxes + len(bbs)

            self.img_ids.append(i+1)
            self.annos.append( bbs)
            img_infos.append(info)
        print ('all images: ', len(self.img_ids))
        print ('bboxes: ' ,  train_bboxes, ignore_bboxes, all_bboxes)
        return img_infos
       

    def get_ann_info(self, idx):
        img_id = self.img_ids[idx]
        ann_info = self.annos[idx]
        return self._parse_ann_info(ann_info)

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
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        ann_list = ann_info
        
        for bb in ann_list:
            #bb = np.array(bb, dtype=np.float32)
            lb = bb[0]
            x1, y1, w, h = bb[1:5]
            x1_vis, y1_vis, w_vis, h_vis = bb[6:10]
            # bbox = [x1, y1, x1+w-1, y1+h-1]
            bbox = [float(x1), float(y1), float(x1)+float(w)-1.0, float(y1)+float(h)-1.0]
            vis_ratio = float(w_vis)*float(h_vis)/float(w)/float(h)
            if lb == 1 and h>=30 and vis_ratio>=0.45:
                gt_bboxes.append( bbox )
                gt_labels.append( lb )
            else:
                gt_bboxes_ignore.append( bbox )
        
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        return ann
