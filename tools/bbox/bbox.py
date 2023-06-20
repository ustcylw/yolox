import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))=}')
import numpy as np
import torch
import copy as Copy





################################################################################
# bbox格式: 
#     xyxycs, <N, 6>
#     xyxy, <N, 4>
################################################################################

def create_bbox(num=0, mode='xyxycs'):
    if num == 0:
        return np.empty((num, len(mode)), dtype=np.float32)
    else:
        return np.zeros((num, len(mode)), dtype=np.float32)


def is_copy(data, copy=True):
    data_ = data
    if copy:
        data_ = Copy.deepcopy(data)
    return data_


def xyxy2xywh(bboxes, copy=True):
    bboxes_ = is_copy(bboxes, copy=copy)
    bboxes_[:, 2:4] -= bboxes_[:, 0:2]
    return bboxes_


def xywh2xyxy(bboxes, copy=True):
    bboxes_ = is_copy(bboxes, copy=copy)
    bboxes_[:, 2:4] += bboxes_[:, 0:2]
    return bboxes_


def xyxy2cxywh(bboxes, copy=True):
    bboxes_ = is_copy(bboxes, copy=copy)
    bboxes_[:, 2:4] -= bboxes_[:, 0:2]
    bboxes_[:, 0:2] -= bboxes_[:, 2:4]/2
    return bboxes_


def cxywh2xyxy(bboxes, copy=True):
    bboxes_ = is_copy(bboxes, copy=copy)
    bboxes_[:, 0:2] += bboxes_[:, 2:4]/2
    bboxes_[:, 2:4] += bboxes_[:, 0:2]
    return bboxes_


def cxywh2xywh(bboxes, copy=True):
    bboxes_ = is_copy(bboxes, copy=copy)
    bboxes_[:, 0:2] += bboxes_[:, 2:4]/2
    return bboxes_


def xywh2cxywh(bboxes, copy=True):
    bboxes_ = is_copy(bboxes, copy=copy)
    bboxes_[:, 0:2] -= bboxes_[:, 2:4]/2
    return bboxes_




def adjust_bboxs_kpts(shape, bboxes, kpts=None, invalid=True):
    h, w = shape
    
    if invalid:
        box_w = bboxes[:, 2] - bboxes[:, 0]
        box_h = bboxes[:, 3] - bboxes[:, 1]
        bboxes = bboxes[np.logical_and(box_w>1, box_h>1)] # discard invalid box
    
    bboxes[bboxes < 0] = 0
    bboxes[:, (0,2)][bboxes[:, (0,2)] > w] = w
    bboxes[:, (0,2)][bboxes[:, (1,3)] > h] = h
    
    if kpts is not None:
        kpts[kpts < 0] = 0
        kpts[:, 0][kpts[:, 0] > w] = w
        kpts[:, 1][kpts[:, 1] > h] = h
    
    return bboxes, kpts