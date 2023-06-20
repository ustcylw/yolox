# /usr/bin/env python
#! coding: utf-8
import os, sys
import cv2
import numpy as np
# from PIL



def shift(img, delta_x, delta_y):
    
    pass


def rotate_center(img, degree, scale, bboxes=None, kpts=None):
    '''
    img: opencv type
    '''
    rows, cols, channel = img.shape
    # 参数：旋转中心 旋转度数 scale
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, scale)
    # 参数：原始图像 旋转参数 元素图像宽高
    rotated = cv2.warpAffine(img, M, (cols, rows))
    bboxes_ = bboxes
    if bboxes_ is not None:
        left, top, right, bottom = bboxes[:, 0:1], bboxes[:, 1:2], bboxes[:, 2:3], bboxes[:, 3:4]
        lt = np.concatenate([left, top], axis=1)
        rt = np.concatenate([right, top], axis=1)
        lb = np.concatenate([left, bottom], axis=1)
        rb = np.concatenate([right, bottom], axis=1)
        r_bboxes = []
        for ps in [lt, rt, lb, rb]:
            xy1 = ps
            xy1 = np.column_stack((xy1.copy(), np.ones(xy1.shape[0]).T))
            xy1 = np.transpose(xy1, (1, 0))
            xy1 = np.dot(M, xy1)
            xy1 = np.transpose(xy1, (1, 0))
            r_bboxes.append(xy1)
        r_bboxes = np.concatenate(r_bboxes, axis=1).reshape((-1, 4, 2))
        l, r, t, b = r_bboxes[:, :, 0].min(axis=1), r_bboxes[:, :, 0].max(axis=1), r_bboxes[:, :, 1].min(axis=1), r_bboxes[:, :, 1].max(axis=1)
        bboxes_ = np.concatenate([l[:, np.newaxis], t[:, np.newaxis], r[:, np.newaxis], b[:, np.newaxis]], axis=1)

    kpts_ = None
    if kpts is not None:
        kpts_ = np.column_stack((kpts.copy(), np.ones(kpts.shape[0]).T))
        kpts_ = np.transpose(kpts_, (1, 0))
        kpts_ = np.dot(M, kpts_)
        kpts_ = np.transpose(kpts_, (1, 0))
    return rotated, bboxes_, kpts_







def rectangle(img, bbox, c=(0, 0, 255)):
    '''
    bbox: <N, 4>
    '''
    for i in range(bbox.shape[0]):
        img = cv2.rectangle(img, bbox[i, 0:2].astype(np.int32), bbox[i, 2:4].astype(np.int32), c, 3)
    return img

def circle(img, kpts):
    for i in range(kpts.shape[0]):
        img = cv2.circle(img, kpts[i, :].astype(np.int32), 5, (255, 0, 0), 3)
    return img

def draw(img, bbox=None, kpts=None):
    if bbox is not None:
        img = rectangle(img, bbox)
    if kpts is not None:
        img = circle(img, kpts)
    return img


def main():
    # rotate_center
    img_file = r'/data/ylw/code/yolo/yolox-pytorch/yolox-v1/data/imgs/002.jpg'
    bbox = np.array([100, 300, 200, 500, 150, 200, 300, 500]).reshape(-1, 4)
    kpts = bbox.copy().reshape((-1, 2)) + 50
    img = cv2.imread(img_file)
    print(f'{img.shape=}')
    img_ori = draw(img.copy(), bbox, kpts)
    cv2.imwrite('./utils/rets/img_ori.jpg', img_ori)
    degree = 90
    scale = 1
    img_, bboxes_, kpts_ = rotate_center(img, degree, scale, bboxes=bbox, kpts=kpts)
    # print(f'{kpts_=}  {bbox.reshape((-1, 2))=}')
    img_ = draw(img_.copy(), bboxes_, kpts=kpts_)
    img_ = rectangle(img_, bbox, c=(255, 255, 0))
    cv2.imwrite(f'./utils/rets/rotate_center.jpg', img_)
    

if __name__ == '__main__':
    
    main()