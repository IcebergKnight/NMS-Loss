import numpy as np
import torch

from . import nms_cuda, nms_cpu
from .soft_nms_cpu import soft_nms_cpu
import copy


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_cuda.nms(dets_th, iou_thr)
        else:
            inds = nms_cpu.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = soft_nms_cpu(
        dets_np,
        iou_thr,
        method=method_codes[method],
        sigma=sigma,
        min_score=min_score)

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)

def tag_nms(dets, tags, tag_thr=0.1):
    dets_tags = torch.cat([dets, tags], dim = 1)
    dets_tags_cpu = dets_tags.cpu().numpy() 
    _, index = tag_nms_cpu(dets_tags_cpu, tag_thr)
    index = dets.new_tensor(index).long()
    return dets[index], index

def tag_nms_cpu(dets, tag_thr=0.1):
    keep = []  
    index = np.arange(len(dets))
    sel_index = []
    while len(dets) > 0:  
        i = np.argmax(dets[:,4])
        keep.append(copy.copy(dets[i]))
        sel_index.append(index[i])
        if len(dets) == 1:
            break
        # swap i and the last one
        dets[i] = dets[-1]
        dets = dets[:-1]
        index[i] = index[-1]
        index = index[:-1]
        # caculate the similarty between i and others
        diff = cacu_l2_diff_cpu(keep[-1], dets)
        inds = np.where(diff >= tag_thr)[0]
        dets = dets[inds]
        index = index[inds]
    return np.array(keep), np.array(sel_index)

def soft_tag_nms(dets, tags, tag_thr=0.1, min_score = 0.1, top_k = -1):
    dets_tags = torch.cat([dets, tags], dim = 1)
    dets_tags_cpu = dets_tags.cpu().numpy()
    dets_tags_cpu, index = soft_tag_nms_cpu(dets_tags_cpu, tag_thr, min_score, top_k)
    index = dets.new_tensor(index).long()
    # dets_tags_cpu = dets.new_tensor(dets_tags_cpu)
    return dets[index], index
	


def soft_tag_nms_cpu(dets, tag_thr=0.1, min_score = 0.1, top_k = -1):
    norm_base = tag_thr
    keep = []  
    index = np.arange(len(dets))
    sel_index = []
    if top_k == -1:
        top_k = len(dets)
    while len(dets) > 0 and len(keep) < top_k:  
        i = np.argmax(dets[:,4])
        keep.append(copy.copy(dets[i]))
        sel_index.append(index[i])
        # swap i and the last one
        dets[i] = dets[-1]
        dets = dets[:-1]
        index[i] = index[-1]
        index = index[:-1]
        # caculate the similarty between i and others
        diff = cacu_l2_diff_cpu(keep[-1], dets)
        inds = np.where(diff < tag_thr)[0]
        diff = diff[inds]
        dets[inds, 4] *= (diff / norm_base)
        # dets[inds, 4] *= np.log(1+(diff / tag_thr * (np.e - 1)))
        sele_inds = np.where(dets[:,4] > min_score)[0]
        dets = dets[sele_inds]
        index = index[sele_inds]
    return np.array(keep), np.array(sel_index)

def cacu_iou_cpu(cur_box, otr_boxes):
    x1 = otr_boxes[:,0]
    y1 = otr_boxes[:,1]
    x2 = otr_boxes[:,2]
    y2 = otr_boxes[:,3]
    area = (x2 - x1) * (y2 - y1)
    xx1 = np.maximum(cur_box[0], x1)
    yy1 = np.maximum(cur_box[1], y1)
    xx2 = np.minimum(cur_box[2], x2)
    yy2 = np.minimum(cur_box[3], y2)
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    iou = inter / ((cur_box[3] - cur_box[1] + 1) * (cur_box[2] - cur_box[0] + 1) + area - inter)
    return iou

def cacu_l2_diff_cpu(cur_box, otr_boxes):
    num = otr_boxes.shape[0]
    tag_dim = otr_boxes.shape[1]
    iou = cacu_iou_cpu(cur_box, otr_boxes)
    cur_box = cur_box[None,:].repeat(num, axis = 0)
    diff = (cur_box[:,5:] - otr_boxes[:,5:])**2
    diff = diff.mean(axis=1)
#     if len(diff) > 0:
#         diff /= np.maximum(np.max(diff), 10)
    iou_zero_idx = iou == 0
    diff[iou_zero_idx] = 99999
    return diff
    

def gen_group_inds(gt_inds):
    target = gt_inds.cpu().numpy()
    gt_label = np.unique(target)
    res = []
    for i in gt_label:
        res.append((np.argwhere(target == i).reshape(-1)))
    return res

def tag_nms_bac(boxes, tags, iou_thr, tag_thr):
    '''
    writern by joeyfang
     args:
         boxes: (Tensor) [n*5] (x1, y1, x2, y2, score)
         tags: (Tensor) [n*tag_dim]
         iou_thr: float
         tag_thr: float
     return: boxes (Tensor) [m*5]
    '''
    # keep = scores.new(scores.size(0)).zero_().long()
    # iou_thr = 0.5
    # check_iou_thr = 0.5
    # tag_thr = 0.5

    keep = boxes.new([]).long()
    if boxes.numel() == 0:
        return boxes[keep]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    num = tags.size()[0]
    size=(num, num, tags.size()[1])
    A = tags.unsqueeze(dim = 1).expand(*size)
    B = A.permute(1, 0, 2)
    diff = A - B
    diff = torch.pow(diff, 2).sum(dim=2)
    cur_diff = boxes.new()

    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep = torch.cat((keep, i.unsqueeze(dim=0)), dim = 0)
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= iou_thr
        check_idx = IoU.ge(iou_thr) & IoU.le(0.8)
        torch.index_select(diff[i], 0, idx, out=cur_diff)
        new_idx = cur_diff.ge(tag_thr)
        # print('****')
        # print(new_idx)
        # print(check_idx)
        # print(new_idx & check_idx)
        # print(IoU.le(iou_thr))
        new_idx = new_idx & check_idx | IoU.le(iou_thr)
        # print(new_idx)
        # new_idx = IoU.le(iou_thr)
        idx = idx[new_idx]

    return torch.cat((boxes[keep], tags[keep]), 1)
    # return boxes[keep]
