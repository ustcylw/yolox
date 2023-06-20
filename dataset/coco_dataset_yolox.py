#! /usr/bin/env python
# coding: utf-8
import os, sys
from copy import deepcopy
import cv2
from PIL import Image
import numpy as np
import copy
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
from random import sample, shuffle

from utils.utils import cvtColor, preprocess_input
from utils.utils_bbox import EncodeBox
from utils.utils_input import InputOPs, ImageIO




import cv2
def rectangle(img, bboxes):
    for box in bboxes:
        img = cv2.rectangle(cv2.UMat(img), box[0:2].astype(np.int32), box[2:4].astype(np.int32), (0, 0, 255), 2)
    return img

def circle(img, pts):
    for pt in pts:
        img = cv2.circle(cv2.UMat(img), pt.astype(np.int32).tolist()[:2], 2, (255, 0, 0), 2)
    return img


class COCODetDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        input_shape, 
        num_classes, 
        mosaic, mosaic_prob, 
        mixup, mixup_prob, 
        train, 
        set_name='train2017', 
        coco=None,
        special_aug_ratio = 0.7,
        mixup_alpha=0.5, mixup_beta=0.5, 
        ex_kpts=0,
        class_list=None
    ) -> None:
        super().__init__()
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        # self.transform = transform
        self.input_shape = input_shape
        self.ex_kpts = ex_kpts

        self.coco      = coco  # COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.classes             = {}
        self.class_list             = class_list
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        self.labels = {}
        self.load_classes()
        
        
        self.num_classes        = len(self.classes)
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_alpha              = mixup_alpha
        self.mixup_beta              = mixup_beta
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.image_ids)
        
        # self.encode_box = EncodeBox(len(self.classes), self.anchors, self.anchors_mask, self.input_shape)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        idx       = idx % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob:
            lines = sample(self.image_ids, 3)
            lines.append(self.image_ids[idx])
            shuffle(lines)
            image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
            if self.mixup and self.rand() < self.mixup_prob:
                lines           = sample(self.image_ids, 1)
                image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box      = self.get_random_data(self.image_ids[idx], self.input_shape, random = self.train)

        # if True:
        #     img_ = rectangle(image.copy(), box[:, 0:4].astype(np.int32))
        #     pts = box[:, 5:].astype(np.int32)
        #     for i in range(pts.shape[0]):
        #         pt = pts[i:i+1, :].reshape(-1, 3)
        #         img_ = circle(img_, pt)
        #     cv2.imwrite(f'./test/imgs/{self.image_ids[idx]}.jpg', img_)

        # image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        image       = np.transpose(InputOPs.norm(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        # print(f'[]  {image.shape=}  {image.min()}/{image.max()}  [===]{box.shape=}  {len(box)=}')
        # 不做[0, 1]
        # if self.ex_kpts > 0 and box.shape[0] > 0:
        #     c, h, w = image.shape
        #     box[:, 5::3] /= w
        #     box[:, 6::3] /= h
        return image, box

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(image_index)[0]
        path       = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        image = Image.open(path)
        image = cvtColor(image)

        return image  # .astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=image_index, iscrowd=False)
        annotations     = np.zeros((0, 5)) if self.ex_kpts < 1 else np.zeros((0, 5+self.ex_kpts*3))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5)) if self.ex_kpts < 1 else np.zeros((1, 5+self.ex_kpts*3))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            if self.ex_kpts > 0:
                annotation[0, 5:] = a['keypoints']
                # print(f'============  {annotation[0, 5:].shape=}  {annotation[0, 5:]=}')
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def num_classes(self):
        # return 80
        return len(self.labels)

    def check_coco(self):
        # print(f'[1]  {len(self.image_ids)=}')
        img_ids = []
        cat_ids = []
        for img_id in self.image_ids:
            ann_id = self.coco.getAnnIds(imgIds=img_id)
            ann = self.coco.loadAnns(ann_id)
            if len(ann) != 0:
                reserve = False
                for a in ann:
                    cat = self.coco.loadCats(a['category_id'])
                    if len(a['bbox']) > 0:
                        if self.class_list is not None:
                            if cat[0]['name'] in self.class_list:
                                if self.ex_kpts > 0:
                                    if a['num_keypoints'] > 0:
                                        reserve = True
                                        cat_ids.append(a['category_id'])
                                else:
                                        reserve = True
                                        cat_ids.append(a['category_id'])
                        else:
                            reserve = True
                            cat_ids.append(a['category_id'])

                if reserve:
                    img_ids.append(img_id)
        self.image_ids = img_ids
        cat_ids.sort()
        self.classes = {}
        self.labels = {}
        for cat_id in set(cat_ids):
            cat = self.coco.loadCats(cat_id)
            self.classes[cat[0]['name']] = len(self.classes)
            self.labels[len(self.classes)-1] = cat[0]['name']
        
        print(f'{self.classes=}')
        print(f'{self.labels=}')
        # print(f'[2]  {len(self.image_ids)=}')



    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image = self.load_image(line)
        # print(f'[get_random_data]  {line=}  {image.size}')
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box = self.load_annotations(line)
        # print(f'[get_random_data]  {line=}  {box.shape=}  {np.array(box)[:, :4].shape=}')

        # img_ = rectangle(np.array(image).copy().astype(np.uint8), box[:, 0:4])
        # if self.ex_kpts > 0:
        #     kpts = box[:, 5:].reshape((-1, 17, 3))#[:, :, 0:2]
        #     # print(f'[get_random_data] ************ {box[:, 5::3].shape}  {kpts.shape=}  {kpts=}')
        #     for kptidx in range(kpts.shape[0]):
        #         img_ = circle(img_, kpts[kptidx, :, :])
        # cv2.imwrite(f'./test/imgs/ori.jpg', img_)

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                if self.ex_kpts > 0:
                    box[:, 5::3] = box[:, 5::3]*nw/iw + dx
                    box[:, 6::3] = box[:, 6::3]*nh/ih + dy
                    box[:, 7::3][box[:, 5::3]>w] = 0
                    box[:, 5::3][box[:, 5::3]>w] = w
                    box[:, 7::3][box[:, 6::3]>h] = 0
                    box[:, 6::3][box[:, 6::3]>h] = h
                    box[:, 7::3][box[:, 5::3]<0] = 0
                    box[:, 5::3][box[:, 5::3]<0] = 0
                    box[:, 7::3][box[:, 6::3]<0] = 0
                    box[:, 6::3][box[:, 6::3]<0] = 0
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            # img_ = rectangle(np.array(image_data).copy().astype(np.uint8), box[:, 0:4])
            # if self.ex_kpts > 0:
            #     kpts = box[:, 5:].reshape((-1, 17, 3))#[:, :, 0:2]
            #     print(f'[get_random_data] ************ {box[:, 5::3].shape}  {kpts.shape=}  {kpts=}')
            #     for kptidx in range(kpts.shape[0]):
            #         img_ = circle(img_, kpts[kptidx, :, :])
            # cv2.imwrite(f'./test/imgs/random.jpg', img_)

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = False #  self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image_data      = np.array(image, np.uint8)  # ******PIL.Image转opencv.image******
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            if self.ex_kpts > 0:
                box[:, 5::3] = box[:, 5::3]*nw/iw + dx
                box[:, 6::3] = box[:, 6::3]*nh/ih + dy
                box[:, 7::3][box[:, 5::3]>w] = 0
                box[:, 5::3][box[:, 5::3]>w] = w
                box[:, 7::3][box[:, 6::3]>h] = 0
                box[:, 6::3][box[:, 6::3]>h] = h
                box[:, 7::3][box[:, 5::3]<0] = 0
                box[:, 5::3][box[:, 5::3]<0] = 0
                box[:, 7::3][box[:, 6::3]<0] = 0
                box[:, 6::3][box[:, 6::3]<0] = 0
                if flip: box[:, 5::3] = w - box[:, 5::3]

            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        # print(f'[get_random_data]  {box.shape=}')
        
        # img_ = rectangle(np.array(image_data).copy().astype(np.uint8), box[:, 0:4])
        # if self.ex_kpts > 0:
        #     kpts = box[:, 5:].reshape((-1, 17, 3))#[:, :, 0:2]
        #     print(f'[get_random_data] ************ {box[:, 5::3].shape}  {kpts.shape=}  {kpts=}')
        #     for kptidx in range(kpts.shape[0]):
        #         img_ = circle(img_, kpts[kptidx, :, :])
        # cv2.imwrite(f'./test/imgs/mosaic.jpg', img_)
        return image_data, box
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                # tmp_box.append(box[4])
                tmp_box = tmp_box + box[4:].tolist()
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = self.load_image(line)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = self.load_annotations(line)
            # print(f'[0]  {len(box)}')
            
            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = False # self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]
                if self.ex_kpts > 0:
                    box[:, 5::3] = iw - box[:, 5::3]

            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            #---------------------------------#
            #   对box进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                
                if self.ex_kpts > 0:
                    box[:, 5::3] = box[:, 5::3]*nw/iw + dx
                    box[:, 6::3] = box[:, 6::3]*nh/ih + dy
                    # box[:, 5::3][box[:, 5::3]>w] = w
                    # box[:, 6::3][box[:, 6::3]>h] = h
                    # box[:, 5::3][box[:, 5::3]<0] = 0
                    # box[:, 6::3][box[:, 6::3]<0] = 0
                    ## ************ 保证点不在图片中，该点v设置为0，而不是给定边界最大值 ************
                    box[:, 7::3][box[:, 5::3]>w] = 0
                    box[:, 5::3][box[:, 5::3]>w] = w
                    box[:, 7::3][box[:, 6::3]>h] = 0
                    box[:, 6::3][box[:, 6::3]>h] = h
                    box[:, 7::3][box[:, 5::3]<0] = 0
                    box[:, 5::3][box[:, 5::3]<0] = 0
                    box[:, 7::3][box[:, 6::3]<0] = 0
                    box[:, 6::3][box[:, 6::3]<0] = 0
                    # if flip: box[:, 5::3] = w - box[:, 5::3]

                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5)) if self.ex_kpts < 1 else np.zeros((len(box), 5+self.ex_kpts*3))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        # print(f'[]  {len(new_boxes)=}')

        return new_image, np.array(new_boxes)

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * self.mixup_alpha + np.array(image_2, np.float32) * self.mixup_beta
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
    
    # DataLoader中collate_fn使用
    def yolox_dataset_collate(self, batch):
        images  = []
        bboxes  = []
        i = 0
        for img, box in batch:
            images.append(img)
            bboxes.append(box)
            i += 1
            
            # img_ = (img.copy().transpose((1,2,0))*255).astype(np.uint8)
            # box[:, 0] -= box[:, 2]/2
            # box[:, 1] -= box[:, 3]/2
            # box[:, 2] += box[:, 0]
            # box[:, 3] += box[:, 1]
            # img_ = rectangle(img_, box[:, 0:4])
            # if self.ex_kpts > 0:
            #     kpts = box[:, 5:].reshape((-1, 17, 3))#[:, :, 0:2]
            #     for kptidx in range(kpts.shape[0]):
            #         img_ = circle(img_, kpts[kptidx, :, :])
            # img_ = cv2.UMat.get(img_)
            # cv2.imwrite(f'/data/ylw/code/yolo/yolox-pytorch/src/test/imgs/collate-{i}.jpg', img_)

        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        return images, bboxes



def rectangle(img, bboxes):
    for box in bboxes:
        img = cv2.rectangle(cv2.UMat(img), box[0:2].astype(np.int32), box[2:4].astype(np.int32), (0, 0, 255), 2)
    return img

def circle(img, pts):
    for pt in pts:
        if pt[2] > 0:
            img = cv2.circle(cv2.UMat(img), pt.astype(np.int32).tolist()[:2], 2, (0, 255, 0), 2)
    return img

