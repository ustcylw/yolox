#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import lightning as L
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, Optional, EPOCH_OUTPUT
from torch.utils.data.dataloader import DataLoader
from PIL import ImageDraw, ImageFont, Image
import colorsys
import tqdm
from torchvision.transforms import ToTensor
import json

from utils.utils_bbox import Decode
from utils.utils import cvtColor, preprocess_input, resize_image
from dataset.coco_dataset_yolox import COCODetDataset
import configs.config_instance as cfg

from tools.pytorch.interface_v3 import EvalModule
from tools.viz.print import show_dict
from net.yolo import YoloBody
from tools.pytorch.device import get_device
from tools.viz.pil_draw import CVDraw, PILDraw

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



import warnings
warnings.filterwarnings("ignore")
import argparse

from model_data.coco_label_map import labelmap
re_labelmap = {str(v):int(k) for k, v in labelmap.items()}


class YoloEvalModule(EvalModule):
    def __init__(self, configs, model) -> None:
        super().__init__(model=model, configs=configs)
        self.configs = configs
        self.device_id = configs.device_id
        self.model = self.load_model(model_file=model, device=self.device_id)
        self.coco_gt = None
        self.input_shape = configs.input_shape
        self.letterbox_image = configs.letterbox_image
        self.decode = Decode(configs=self.configs)

    def load_model(self, model_file, device):
        print(f'loading model: {model_file} ...')
        if isinstance(model_file, str):
            model = YoloBody(self.configs.num_classes, phi=self.configs.phi, ex_kpts=self.configs.ex_kpts)
            # model = model.load(model_file, mode=None, model=model, device=device, strict=True)
            model = model.load_checkpoints(model_file, mode='state_dict', device=device, strict=True)
        else:
            model = model_file
        model.eval()
        model.to(get_device(device))
        print(f'load model {model_file} complete.')
        return model

    def pred_image(self, image, model=None, device=0, params=None, draw=True):
        self.pred_model = self.model
        if isinstance(model, str):
            print(f'loading new model {model} ...')
            self.pred_model = self.load_model(model, device)
            print('{} model, and classes loaded.'.format(model))
        self.pred_model.eval()

        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.configs.input_shape[1], self.configs.input_shape[0]), False)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.device_id > -1:
                images = images.to(get_device(device))
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.pred_model(images)
        
        det_infos = self.decode.decode(outputs, image)
        if self.configs.ex_kpts > 0:
            image, bboxes, labels, top_kpts = det_infos
        else:
            image, bboxes, labels = det_infos

        if draw and bboxes.size != 0:
            bboxes_ = bboxes[:, 0:4].copy()
            bboxes_[:, 2:4] += bboxes_[:, 0:2]
            labels_ = [{'label': label[0], 'score': np.round(bbox[5], 2), 'cls': int(bbox[4])} for bbox, label in zip(bboxes, labels)]
            # image_ = CVDraw(np.array(image)).rectangle(
            #     np.array(image), 
            #     bboxes_, 
            #     labels=labels_, 
            #     center=True, corner=False
            # )
            # image = Image.fromarray(image_)
            image = PILDraw().rectangle(
                image, 
                bboxes_, 
                labels=labels_, 
            )

        return image, bboxes, labels

    def pred_image_1(self, image, model=None, device=0, params=None, draw=True, crop = False, count = False):
        self.pred_model = self.model
        if isinstance(model, str):
            print(f'loading new model {model} ...')
            self.pred_model = self.load_model(model, device)
            print('{} model, and classes loaded.'.format(model))
        self.pred_model.eval()

        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.configs.input_shape[1], self.configs.input_shape[0]), self.configs.letterbox_image)
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.device_id > -1:
                images = images.to(get_device(device))
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.pred_model(images)

        det_infos = self.decode.decode(outputs, image)
        if self.configs.ex_kpts > 0:
            image, bboxes, labels, top_kpts = det_infos
        else:
            image, bboxes, labels = det_infos

        if draw and bboxes.size != 0:
            bboxes_ = bboxes[:, 0:4]
            bboxes_[:, 2:4] += bboxes_[:, 0:2]
            labels_ = [{'label': label[0], 'score': np.round(bbox[5], 2), 'cls': int(bbox[4])} for bbox, label in zip(bboxes, labels)]
            # image_ = CVDraw(np.array(image)).rectangle(
            #     np.array(image), 
            #     bboxes_, 
            #     labels=labels_, 
            #     center=True, corner=False
            # )
            # image = Image.fromarray(image_)
            image = PILDraw().rectangle(
                image, 
                bboxes_, 
                labels=labels_, 
            )
        return image, bboxes, labels

    def pred_coco(self, root_dir, set_name='train2017', output_dir='./', save_image=True):
        ann_file = os.path.join(root_dir, 'annotations', 'instances_' + set_name + '.json')
        self.coco_gt = COCO(annotation_file=ann_file)
        img_ids = self.coco_gt.getImgIds()
        
        results = []
        ret_img_ids = []
        tmp_str = ''
        for idx, img_id in enumerate(tqdm.tqdm(img_ids, desc='Predict: ', unit=' img', leave=True, file=sys.stdout, ncols=100, position=0, colour='blue')):
            imgs = self.coco_gt.loadImgs(img_id)
            img_name = imgs[0]['file_name']
            img_file = os.path.join(root_dir, set_name, img_name)
            
            try:
                image = Image.open(img_file)
            except:
                print('Open Error! Try again!')
                continue
            else:
                image, bboxes, labels = self.pred_image(image=image, device=self.device_id, draw=save_image)
            if bboxes is None: #  or clses is None:
                continue
            ret_img_ids.append(img_id)
            for bbox in bboxes:
                cid = re_labelmap[f'{int(bbox[4]+1)}']
                results.append({
                    'image_id': img_id, 
                    # 'category_id': int(bbox[4]+1), # ****** 注意：这里一定要+1，不然map计算非常低，因为coco计算map时，category_id是从1开始的 ******
                    'category_id': cid, # ****** 注意：这里一定要+1，不然map计算非常低，因为coco计算map时，category_id是从1开始的 ******
                    'score': round(float(bbox[5]), 4), 
                    'bbox': [round(float(bbox[0]), 2), round(float(bbox[1]), 2), round(float(bbox[2]), 2), round(float(bbox[3]), 2)]
                })
                tmp_str += str(img_id) + ' ' + str(int(bbox[4])) + ' ' + str(float(bbox[5])) + ' ' + str(round(float(bbox[0]), 3)) + ' ' + str(round(float(bbox[1]), 3)) + ' ' + str(round(float(bbox[2]), 3)) + ' ' + str(round(float(bbox[3]), 3)) + '\n'
        
            if save_image:
                save_image_path = os.path.join(output_dir, 'images', img_name)
                image.save(save_image_path)
        pred_ret_file = os.path.join(output_dir, f'pre-{set_name}.json')
        with open(pred_ret_file, 'w') as df:
            json.dump(results, df, indent=4)
        img_ids_str = ' '.join([str(img_id) for img_id in img_ids])
        with open(os.path.join(output_dir, f'pre-{set_name}.txt'), 'w') as df:
            df.writelines(img_ids_str)
        
        with open('./tmp_str.txt', 'w') as df:
            df.writelines(tmp_str)
            
        return self.coco_gt, results, ret_img_ids

    def COCOmAP(self, coco_gt, coco_dt, img_ids, iou_type='bbox'):
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
        # coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    def onnx(self, model=None, input=None, save_file=None, input_names=['input'], output_names=['output']):
        print(f'to onnx ...')
        model_ = self.model if model is None else model
        model_.eval()
        batch_size = 1
        input_ = torch.randn(
            batch_size, 
            3, 
            self.configs.input_shape[0], 
            self.configs.input_shape[1], 
            requires_grad=True,
            device=get_device(self.configs.device_id)
        )
        if input is not None:
            input_ = input
        save_file_ = save_file if save_file is not None else self.configs.onnx_save_file
        output_ = model_(input_)
        torch.onnx.export(model_,        # 模型的名称
                  input_,   # 一组实例化输入
                  save_file_,   # 文件保存路径/名称
                  export_params=True,        #  如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                  opset_version=10,          # ONNX 算子集的版本，当前已更新到15
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names = input_names,   # 输入模型的张量的名称
                  output_names = output_names, # 输出模型的张量的名称
                  # dynamic_axes将batch_size的维度指定为动态，
                  # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
                  dynamic_axes={'input' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})
        print(f'to onnx complete.')

    
    def test(self):
        gt_ann_file = r'/data/ylw/dataset/coco/annotations//instances_val2017.json'
        dt_ann_file = r'/home/yangliwei/code/yolo/yolov5_pl/val_results/pre-val2017_gt_rand_bbox.json'
        coco_gt = COCO(annotation_file=gt_ann_file)
        coco_dt = coco_gt.loadRes(dt_ann_file)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        maxdet_index = coco_eval.params.maxDets.index(100)
        coco_eval.eval['precision'][iou_index, :, :, area_index, maxdet_index].mean()
        
        return coco_eval.stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')

    # parser.add_argument('--configs', type=str, default=r'./configs/config_instance_s.py', help='configs path')
    parser.add_argument('--configs', type=str, default=r'./configs/config_instance.py', help='configs path')
    args = parser.parse_args()
    import imp
    cfg = imp.load_source('b', args.configs)
    # print(f'>'*80)
    show_dict(cfg.EVALCONFIGS)
    # print(f'<'*80)
    CONFIGS = cfg.EVALCONFIGS
    # print(f'============  {args.configs}  {cfg=}')

    mode = CONFIGS.mode
    print(f'{CONFIGS=}  {cfg=}  {CONFIGS.mode=}  {CONFIGS.model_path=}')
    eval_module = YoloEvalModule(configs=CONFIGS, model=CONFIGS.model_path)
    
    if mode == 'coco_map':
        # root_dir = r'/data8022/ylw/dataset/voc2coco'
        # set_name = 'val'
        transformer = [ToTensor()]
        output_dir = './exps/runs/eval_dir'
        gt_ann_file = os.path.join(CONFIGS.dataset_root, 'annotations', 'instances_' + CONFIGS.dataset_set_name + '.json')
        dt_ann_file = os.path.join(output_dir, f'pre-{CONFIGS.dataset_set_name}.json')
        if not os.path.exists(dt_ann_file):
            coco_gt, _, img_ids = eval_module.pred_coco(root_dir=CONFIGS.dataset_root, set_name=CONFIGS.dataset_set_name, output_dir=output_dir, save_image=True)
        else:
            coco_gt = COCO(gt_ann_file)
            img_ids = None
            with open(dt_ann_file.replace('json', 'txt'), 'r') as lf:
                img_ids = lf.readlines()[0].strip().split(' ')
            img_ids = [int(img_id) for img_id in img_ids]
                
        coco_pred = coco_gt.loadRes(os.path.join(output_dir, f'pre-{CONFIGS.dataset_set_name}.json'))
        # run coco eval
        eval_module.COCOmAP(coco_gt=coco_gt, coco_dt=coco_pred, img_ids=img_ids, iou_type='bbox')
    elif mode == 'onnx':
        eval_module.onnx()
    elif mode == 'test':
        eval_module.test()
        