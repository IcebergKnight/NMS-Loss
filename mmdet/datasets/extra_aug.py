import mmcv
import numpy as np
import copy
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, all_bboxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, all_bboxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, all_bboxes, labels):
        if random.randint(2):
            return img, all_bboxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        for bboxes in all_bboxes.values():
            boxes += np.tile((left, top), 2)
        return img, all_bboxes, labels


class RandomCropA(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, all_bboxes, labels):
        h, w, c = img.shape
        
        # 1. read in
        bboxes = all_bboxes['bboxes']
        if 'bboxes_ignore' in all_bboxes:
            bboxes_ignore = all_bboxes['bboxes_ignore']

        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, all_bboxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    bboxes.reshape(-1, 4), patch.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                bboxes = bboxes[mask]
                labels = labels[mask]

                # 2. adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                bboxes[:, 2:] = bboxes[:, 2:].clip(max=patch[2:])
                bboxes[:, :2] = bboxes[:, :2].clip(min=patch[:2])
                bboxes -= np.tile(patch[:2], 2)

                if 'bboxes_ignore' in all_bboxes:
                    bboxes_ignore[:, 2:] = bboxes_ignore[:, 2:].clip(max=patch[2:])
                    bboxes_ignore[:, :2] = bboxes_ignore[:, :2].clip(min=patch[:2])
                    bboxes_ignore -= np.tile(patch[:2], 2)
                    bboxes_ignore = bboxes_ignore.clip(min=[0, 0, 0, 0]).astype(np.float32)
                
                # 3. write in
                all_bboxes['bboxes'] = bboxes
                if 'bboxes_ignore' in all_bboxes:
                    all_bboxes['bboxes_ignore'] = bboxes_ignore

                return img, all_bboxes, labels


class RandomCropB(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious)
        self.min_crop_size = min_crop_size

    def __call__(self, img, all_bboxes, labels):
        h, w, c = img.shape
        
        # 1. read in
        bboxes = all_bboxes['bboxes']
        if 'bboxes_ignore' in all_bboxes:
            bboxes_ignore = all_bboxes['bboxes_ignore']

        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, all_bboxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    bboxes.reshape(-1, 4), patch.reshape(-1, 4), mode='iof').reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                bboxes = bboxes[mask]
                labels = labels[mask]

                # 2. adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                bboxes[:, 2:] = bboxes[:, 2:].clip(max=patch[2:])
                bboxes[:, :2] = bboxes[:, :2].clip(min=patch[:2])
                bboxes -= np.tile(patch[:2], 2)

                if 'bboxes_ignore' in all_bboxes:
                    bboxes_ignore[:, 2:] = bboxes_ignore[:, 2:].clip(max=patch[2:])
                    bboxes_ignore[:, :2] = bboxes_ignore[:, :2].clip(min=patch[:2])
                    bboxes_ignore -= np.tile(patch[:2], 2)
                    bboxes_ignore = bboxes_ignore.clip(min=[0, 0, 0, 0]).astype(np.float32)
                
                # 3. write in
                all_bboxes['bboxes'] = bboxes
                if 'bboxes_ignore' in all_bboxes:
                    all_bboxes['bboxes_ignore'] = bboxes_ignore

                return img, all_bboxes, labels


class RandomCropC(object):

    def __init__(self,
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, all_bboxes, labels):
        h, w, c = img.shape
        
        # 1. read in
        bboxes = all_bboxes['bboxes']
        if 'bboxes_ignore' in all_bboxes:
            bboxes_ignore = all_bboxes['bboxes_ignore']

        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, all_bboxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))

                # center of boxes should inside the crop img
                center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                bboxes = bboxes[mask]
                labels = labels[mask]

                # 2. adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                bboxes[:, 2:] = bboxes[:, 2:].clip(max=patch[2:])
                bboxes[:, :2] = bboxes[:, :2].clip(min=patch[:2])
                bboxes -= np.tile(patch[:2], 2)

                if 'bboxes_ignore' in all_bboxes:
                    bboxes_ignore[:, 2:] = bboxes_ignore[:, 2:].clip(max=patch[2:])
                    bboxes_ignore[:, :2] = bboxes_ignore[:, :2].clip(min=patch[:2])
                    bboxes_ignore -= np.tile(patch[:2], 2)
                    bboxes_ignore = bboxes_ignore.clip(min=[0, 0, 0, 0]).astype(np.float32)
                
                # 3. write in
                all_bboxes['bboxes'] = bboxes
                if 'bboxes_ignore' in all_bboxes:
                    all_bboxes['bboxes_ignore'] = bboxes_ignore

                return img, all_bboxes, labels


class RandomCropSplice(object):

    def __init__(self,
                 crop_size=0.5):
        # 1: return ori img
        self.crop_size = crop_size

    def gen_noempty_patch(self, w, bboxes, bboxes_ignore, labels, img, base_w):
        while True:
            for i in range(50):
                new_w = self.crop_size * w

                left = random.uniform(0, w - new_w)

                patch = np.array((int(left), int(left + new_w)))

                if len(bboxes) != 0:
                    # center of boxes should inside the crop img
                    center = (bboxes[:, 0] + bboxes[:, 2]) / 2
                    mask = (center > patch[0]) * (center < patch[1])
                    if not mask.any():
                        continue
                    bboxes = bboxes[mask]
                    labels = labels[mask]

                    # 2. adjust boxes
                    img = img[:, patch[0]:patch[1]]
                    bboxes[:,0] = bboxes[:,0] - patch[0] + base_w
                    bboxes[:,2] = bboxes[:,2] - patch[0] + base_w

                if bboxes_ignore is not None:
                    # bboxes_ignore[:, 2:] = bboxes_ignore[:, 2:].clip(max=patch[2:])
                    # bboxes_ignore[:, :2] = bboxes_ignore[:, :2].clip(min=patch[:2])
                    bboxes_ignore[:,0] = bboxes_ignore[:,0] - patch[0] + base_w
                    bboxes_ignore[:,2] = bboxes_ignore[:,2] - patch[0] + base_w
                    bboxes_ignore = bboxes_ignore.clip(min=[0, 0, 0, 0]).astype(np.float32)
                
                return img, bboxes, bboxes_ignore, labels
        

    def __call__(self, img1, all_bboxes1, labels1,
                       img2, all_bboxes2, labels2):
        h, w, c = img1.shape
        # 1. read in
        bboxes1 = all_bboxes1['bboxes']
        bboxes2 = all_bboxes2['bboxes']
        #stamp = random.uniform(0, 1)
        #print(stamp, all_bboxes)
        bboxes_ignore1 = None
        bboxes_ignore2 = None
        if 'bboxes_ignore' in all_bboxes1:
            bboxes_ignore1 = all_bboxes1['bboxes_ignore']
        if 'bboxes_ignore' in all_bboxes2:
            bboxes_ignore2 = all_bboxes2['bboxes_ignore']
        img1_patch, bboxes1_patch, bboxes_ignore1_patch, labels1_patch = \
            self.gen_noempty_patch(w, bboxes1, bboxes_ignore1, labels1, img1, 0)
        img2_patch, bboxes2_patch, bboxes_ignore2_patch, labels2_patch = \
            self.gen_noempty_patch(w, bboxes2, bboxes_ignore2, labels2, img2, w * self.crop_size)
        img = np.concatenate([img1_patch, img2_patch], axis = 1)
        bboxes = np.concatenate([bboxes1_patch, bboxes2_patch], axis = 0)
        labels = np.concatenate([labels1_patch, labels2_patch], axis = 0)
        if bboxes_ignore1_patch is not None and bboxes_ignore2_patch is not None:
            bboxes_ignore = np.concatenate([bboxes_ignore1_patch, bboxes_ignore2_patch], axis = 0)
        else:
            bboxes_ignore = bboxes_ignore1_patch if bboxes_ignore1_patch is not None else bboxes_ignore2_patch
        all_bboxes = {}
        all_bboxes['bboxes'] = bboxes
        if bboxes_ignore is not None:
            all_bboxes['bboxes_ignore'] = bboxes_ignore

        return img, all_bboxes, labels


class RandomCropSpliceHori(object):

    def __init__(self,
                 crop_size=0.5):
        # 1: return ori img
        self.crop_size = crop_size

    def gen_noempty_patch(self, h, bboxes, bboxes_ignore, labels, img, base_h):
        while True:
            for i in range(50):
                new_h = self.crop_size * h

                top = random.uniform(0, h - new_h)

                patch = np.array((int(top), int(top + new_h)))

                if len(bboxes) != 0:
                    # center of boxes should inside the crop img
                    center = (bboxes[:, 1] + bboxes[:, 3]) / 2
                    mask = (center > patch[0]) * (center < patch[1])
                    if not mask.any():
                        continue
                    bboxes = bboxes[mask]
                    labels = labels[mask]

                    # 2. adjust boxes
                    img = img[patch[0]:patch[1], :]
                    bboxes[:,1] = bboxes[:,1] - patch[0] + base_h
                    bboxes[:,3] = bboxes[:,3] - patch[0] + base_h

                if bboxes_ignore is not None:
                    # bboxes_ignore[:, 2:] = bboxes_ignore[:, 2:].clip(max=patch[2:])
                    # bboxes_ignore[:, :2] = bboxes_ignore[:, :2].clip(min=patch[:2])
                    bboxes_ignore[:,1] = bboxes_ignore[:,1] - patch[0] + base_h
                    bboxes_ignore[:,3] = bboxes_ignore[:,3] - patch[0] + base_h
                    bboxes_ignore = bboxes_ignore.clip(min=[0, 0, 0, 0]).astype(np.float32)
                
                return img, bboxes, bboxes_ignore, labels
        

    def __call__(self, img1, all_bboxes1, labels1,
                       img2, all_bboxes2, labels2):
        h, w, c = img1.shape
        # 1. read in
        bboxes1 = all_bboxes1['bboxes']
        bboxes2 = all_bboxes2['bboxes']
        #stamp = random.uniform(0, 1)
        #print(stamp, all_bboxes)
        bboxes_ignore1 = None
        bboxes_ignore2 = None
        if 'bboxes_ignore' in all_bboxes1:
            bboxes_ignore1 = all_bboxes1['bboxes_ignore']
        if 'bboxes_ignore' in all_bboxes2:
            bboxes_ignore2 = all_bboxes2['bboxes_ignore']
        img1_patch, bboxes1_patch, bboxes_ignore1_patch, labels1_patch = \
            self.gen_noempty_patch(h, bboxes1, bboxes_ignore1, labels1, img1, 0)
        img2_patch, bboxes2_patch, bboxes_ignore2_patch, labels2_patch = \
            self.gen_noempty_patch(h, bboxes2, bboxes_ignore2, labels2, img2, h * self.crop_size)
        img = np.concatenate([img1_patch, img2_patch], axis = 0)
        bboxes = np.concatenate([bboxes1_patch, bboxes2_patch], axis = 0)
        labels = np.concatenate([labels1_patch, labels2_patch], axis = 0)
        if bboxes_ignore1_patch is not None and bboxes_ignore2_patch is not None:
            bboxes_ignore = np.concatenate([bboxes_ignore1_patch, bboxes_ignore2_patch], axis = 0)
        else:
            bboxes_ignore = bboxes_ignore1_patch if bboxes_ignore1_patch is not None else bboxes_ignore2_patch
        all_bboxes = {}
        all_bboxes['bboxes'] = bboxes
        if bboxes_ignore is not None:
            all_bboxes['bboxes_ignore'] = bboxes_ignore

        return img, all_bboxes, labels
class RandomCropD(object):

    def __init__(self,
                 crop_size=0.5):
        # 1: return ori img
        self.crop_size = crop_size

    def __call__(self, img, all_bboxes, labels):
        h, w, c = img.shape

        # 1. read in
        bboxes = all_bboxes['bboxes']
        #stamp = random.uniform(0, 1)
        #print(stamp, all_bboxes)
        if 'bboxes_ignore' in all_bboxes:
            bboxes_ignore = all_bboxes['bboxes_ignore']

        while True:
            for i in range(50):
                new_w = self.crop_size * w
                new_h = self.crop_size * h

                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))

                if len(bboxes) != 0:
                    # center of boxes should inside the crop img
                    center = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                    mask = (center[:, 0] > patch[0]) * (
                        center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                            center[:, 1] < patch[3])
                    if not mask.any():
                        continue
                    bboxes = bboxes[mask]
                    labels = labels[mask]

                    # 2. adjust boxes
                    # bboxes[:, 2:] = bboxes[:, 2:].clip(max=patch[2:])
                    # bboxes[:, :2] = bboxes[:, :2].clip(min=patch[:2])
                    bboxes -= np.tile(patch[:2], 2)

                if 'bboxes_ignore' in all_bboxes:
                    # bboxes_ignore[:, 2:] = bboxes_ignore[:, 2:].clip(max=patch[2:])
                    # bboxes_ignore[:, :2] = bboxes_ignore[:, :2].clip(min=patch[:2])
                    bboxes_ignore -= np.tile(patch[:2], 2)
                    bboxes_ignore = bboxes_ignore.clip(min=[0, 0, 0, 0]).astype(np.float32)
                
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                # 3. write out
                all_bboxes['bboxes'] = bboxes
                if 'bboxes_ignore' in all_bboxes:
                    all_bboxes['bboxes_ignore'] = bboxes_ignore
                #print(stamp, all_bboxes)
                return img, all_bboxes, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop_a=None,
                 random_crop_b=None,
                 random_crop_c=None,
                 random_crop_d=None,
                 random_crop_splice=None,
                 random_crop_splice_hori=None):
        self.transforms = []

        if expand is not None:
            self.transforms.append(Expand(**expand))

        # original
        if random_crop_a is not None:
            self.transforms.append(RandomCropA(**random_crop_a))
        # iou -> iof
        if random_crop_b is not None:
            self.transforms.append(RandomCropB(**random_crop_b))
        # no iou filter
        if random_crop_c is not None:
            self.transforms.append(RandomCropC(**random_crop_c))
        # netw, neth size is fixed
        if random_crop_d is not None:
            self.transforms.append(RandomCropD(**random_crop_d))
        # splice two image
        if random_crop_splice is not None:
            self.transforms.append(RandomCropSplice(**random_crop_splice))
        # splice two image in horizeontel
        if random_crop_splice_hori is not None:
            self.transforms.append(RandomCropSpliceHori(**random_crop_splice_hori))

        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))

    def __call__(self, img, boxes, labels,
            add_img = None, add_boxes = None, add_labels = None):
        img = img.astype(np.float32)
        if add_img is not None:
            add_img = add_img.astype(np.float32)
        for transform in self.transforms:
            if add_img is None:
                img, boxes, labels = transform(img, boxes, labels)
            else:
                img, boxes, labels = transform(img, boxes, labels,
                        add_img, add_boxes, add_labels)
                add_img = None
        return img, boxes, labels


