import os, sys
sys.path.append(r'/data/ylw/code/yolo/yolox-pytorch/src')

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from net.yolo import YoloBody



def test_net():
    
    num_classes=1
    phi='s'
    ex_kpts=17
    
    
    net = YoloBody(num_classes, phi, ex_kpts)
    
    x = torch.ones(size=(2, 3, 640, 640))
    y = net(x)
    for idx, yi in enumerate(y):
        print(f'[{idx}]  {yi.shape=}')



if __name__ == '__main__':
    test_net()