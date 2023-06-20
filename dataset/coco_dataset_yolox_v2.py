#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
import albumentations as A


from utils.utils import cvtColor, preprocess_input
# from utils.utils_bbox import EncodeBox
from utils.utils_input import InputOPs, ImageIO
from tools.augs.augs import rotate_center_keep
from tools.image import image as TImage
from tools.augs import augs as Augs 
import tools.bbox.bbox as TBbox

from tools.viz.pil_draw import PILDraw



class COCODetDataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        input_shape, 
        mosaic, mosaic_prob, 
        mixup, mixup_prob, 
        train, 
        set_name='train2017', 
        coco=None,
        special_aug_ratio = 0.7,
        mixup_alpha=0.5, mixup_beta=0.5, 
        jitter=True, random_scale=0,
        ex_kpts=0,
        class_list=None, 
        trans=None
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
        self.jitter = jitter
        self.random_scale = random_scale

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
        
        self.trans = None # self.get_transforms() if trans is None else trans
        
        # self.encode_box = EncodeBox(len(self.classes), self.anchors, self.anchors_mask, self.input_shape)

    def get_transforms(self, bboxes=None, kpts=None):
        bbox_params = None
        if bboxes is not None and bboxes.shape[0] > 0:
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes'])
            # bbox_params=A.BboxParams(format='coco', label_fields=['bbox_classes'])
        keypoint_params=None
        if self.ex_kpts > 0 and kpts is not None and kpts.shape[0] > 0:
            keypoint_params=A.KeypointParams(format='xy', label_fields=['kpts_classes'])
            
        trans = A.Compose(
            [
                A.OneOf(
                    [
                        A.NoOp(p=0.5), 
                        A.ColorJitter(brightness=0.3, hue=0.3, saturation=0.3, contrast=0.3, p=0.5)
                    ]
                ), 
                A.SafeRotate(np.random.randint(45, 90), border_mode=cv2.BORDER_CONSTANT, value=(114.0,114.0,114.0), p=0.5), 
                A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=(114.0,114.0,114.0), p=1.0), 
                A.Resize(height=640, width=640, p=1.0)
            ], 
            bbox_params=bbox_params, 
            keypoint_params=keypoint_params
        )
        return trans
    
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

    def get_random_item(self):
        return self.image_ids[np.random.randint(0, self.length)%self.length]
    
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

        image       = np.transpose(InputOPs.norm(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            
        # 不做[0, 1]
        # if self.ex_kpts > 0 and box.shape[0] > 0:
        #     c, h, w = image.shape
        #     box[:, 5::3] /= w
        #     box[:, 6::3] /= h
        return image, box

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(image_index)[0]
        path       = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        # image = Image.open(path)
        # image = cvtColor(image)
        image = cv2.imread(path)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # print(f'[load_image]  ===============  {image.shape=}')

        return image

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

    # def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.9, sat=0.2, val=0.4, random=True):
        line    = annotation_line
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image = self.load_image(line)
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box = self.load_annotations(line)            

        bboxes, bboxes_labels = None, None
        kpts, kpts_labels = None, None
        if box.shape[0] > 0:
            bboxes = box[:, 0:4]
            # bboxes_labels = np.array([str(int(bl)) for bl in box[:, 4:5]])
            bboxes_labels = np.array([int(bl) for bl in box[:, 4:5]])
            if self.ex_kpts > 0:
                # TODO
                kpts = None
                kpts_labels = None
        
            trans = self.get_transforms(bboxes=bboxes, kpts=kpts)
            trans_data = trans(image=image, bboxes=bboxes, bbox_classes=bboxes_labels, keypoints=kpts, kpts_classes=kpts_labels)
            image, bboxes, bboxes_labels, kpts, kpts_labels = trans_data['image'], np.array(trans_data['bboxes']), np.array(trans_data['bbox_classes']), np.array(trans_data['keypoints']), np.array(trans_data['kpts_classes'])
            bboxes_labels = np.array([float(bl) for bl in bboxes_labels])
            bboxes = np.concatenate([bboxes, bboxes_labels[:, np.newaxis]], axis=1)
        else:
            trans = self.get_transforms(bboxes=None, kpts=None)
            trans_data = trans(image=image)
            image = trans_data['image']
            bboxes = box
            
        return image, bboxes
    
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
        for line in annotation_line:
            # self.get_random_data
            image, bboxes = self.get_random_data(line, input_shape, jitter, hue, sat, val)
            image_datas.append(image)
            box_datas.append(bboxes)
            
        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3], dtype=np.uint8)
        new_patchs = [
            Image.fromarray(new_image[:cuty, :cutx, :]), 
            Image.fromarray(new_image[cuty:, :cutx, :]),
            Image.fromarray(new_image[cuty:, cutx:, :]),
            Image.fromarray(new_image[:cuty, cutx:, :])
        ]
        # start_xy = [
        #     [slice(0, cuty), slice(0, cutx)], 
        #     [slice(cuty, None), slice(0, cutx)], 
        #     [slice(cuty, None), slice(cutx, None)], 
        #     [slice(0, cuty), slice(cutx, None)]
        # ]
        start_xy = [
            [slice(0, cuty), slice(0, cutx)], 
            [slice(cuty, 640), slice(0, cutx)], 
            [slice(cuty, 640), slice(cutx, 640)], 
            [slice(0, cuty), slice(cutx, 640)]
        ]
        
        # for patch_idx, (patch, xy, image_data) in enumerate(zip(new_patchs, start_xy, image_datas)):
        #     bboxes, kpts = None, None
        #     if box_datas[patch_idx].shape[0] > 0:
        #         bboxes = box_datas[patch_idx][:, 0:4]
        #         if self.ex_kpts > 0:
        #             kpts = box_datas[patch_idx][:, 5:].reshape(-1,3)  # TODO
        #     img1, bboxes, kpts = Augs.paste(patch, Image.fromarray(image_data), bboxes, kpts)
        #     new_image[xy[0], xy[1], :] = np.array(img1)
        #     if box_datas[patch_idx].shape[0] > 0:
        #         bboxes[:, (0, 2)] += xy[1].start
        #         bboxes[:, (1, 3)] += xy[0].start
        #         box_datas[patch_idx][:, 0:4] = bboxes
        #         if self.ex_kpts:
        #             # TODO
        #             ...
        for patch_idx, (patch, xy, image_data) in enumerate(zip(new_patchs, start_xy, image_datas)):
            bboxes, kpts = None, None
            if box_datas[patch_idx].shape[0] > 0:
                bboxes = box_datas[patch_idx][:, 0:4]
                bboxes_labels = np.array([str(int(bl)) for bl in box_datas[patch_idx][:, 4:5]])
                if self.ex_kpts > 0:
                    kpts = box_datas[patch_idx][:, 5:].reshape(-1,3)  # TODO
                    
                trans = A.Compose(
                    [A.Resize(height=xy[0].stop-xy[0].start, width=xy[1].stop-xy[1].start, p=1.0)], 
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes'])
                )
                trans_data = trans(image=image_data, bboxes=bboxes, bbox_classes=bboxes_labels)
                image, bboxes, bboxes_labels = trans_data['image'], np.array(trans_data['bboxes']), np.array(trans_data['bbox_classes'])
                
                new_image[xy[0], xy[1], :] = image
                if box_datas[patch_idx].shape[0] > 0:
                    bboxes[:, (0, 2)] += xy[1].start
                    bboxes[:, (1, 3)] += xy[0].start
                    box_datas[patch_idx][:, 0:4] = bboxes
                    if self.ex_kpts:
                        # TODO
                        ...

        return new_image, np.concatenate(box_datas, axis=0)

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
        for img, box in batch:
            images.append(img)
            bboxes.append(box)
            
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

        images  = torch.from_numpy(np.array(images))
        images = images.type(torch.FloatTensor)
        bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        
        return images, bboxes

