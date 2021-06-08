import numpy as np
import scipy.io
import os
import sys
import cv2
import pickle

def parse_pred(pred_path):
    pred = pickle.load(open(pred_path, 'rb'))
    return pred
                       
def parse_ann_name(label_path):
    ann = pickle.load(open(label_path, 'rb'))
    img_infos=ann['img_infos']
    file_names = [info['filename'] for info in img_infos]
    return np.array(file_names)

def gen_txt(preds, names, min_score=0.5, min_height = 50):
    file_pointers = {}
    count = len(preds)
    print('total count {}'.format(str(count)))
    for i in range(count):
        name = names[i]
        name_splits = name.split('_')
        _set = name_splits[0]
        _video = name_splits[1] + '.txt'
        _img = int(name_splits[2][1:].split('.')[0]) + 1
        dir_path = os.path.join(save_path, _set)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fp_path = os.path.join(dir_path, _video)
        fp = None
        if fp_path in file_pointers.keys():
            fp = file_pointers[fp_path]
        else:
            fp = open(fp_path, 'w')
            print('writing {}'.format(fp_path))
            file_pointers[fp_path] = fp
        if preds[i] is None:
            continue
        pred = preds[i][0]
        if len(pred) == 0:
            continue
        pred = np.array(pred)
        pred[np.where(pred[:,4] < min_score),4] = 0
        pred[:,2] = pred[:,2] - pred[:,0] + 1
        pred[:,3] = pred[:,3] - pred[:,1] + 1
        if min_height > 0:
            pred = pred[np.where(pred[:,3] >= min_height)]
        for p in pred:
            row = [_img] + p.tolist()
            row = ",".join(str(item) for item in row)
            row = row+'\n'
            fp.write(row)
    for key in file_pointers.keys():
        fp = file_pointers[key]
        fp.close()

pred_path = 'results/caltech.pkl'
save_path = 'results/NMS-Ped_Caltech'
label_path = "/data/research/caltech/annotations/caltech_test.pkl"

preds = parse_pred(pred_path)
names = parse_ann_name(label_path)
gen_txt(preds, names, -1, -1)