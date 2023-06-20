import os, sys
sys.path.append(r'/data/ylw/code/yolo/yolox-pytorch/src_test_1')

import cv2
import numpy as np
from pycocotools.coco import COCO

from dataset.coco_dataset_yolox import COCODetDataset
import time





def rectangle(img, bboxes):
    for box in bboxes:
        img = cv2.rectangle(cv2.UMat(img), box[0:2].astype(np.int32), box[2:4].astype(np.int32), (0, 0, 255), 2)
    return img

def circle(img, pts):
    for pt in pts:
        img = cv2.circle(cv2.UMat(img), pt.astype(np.int32).tolist()[:2], 2, (255, 0, 0), 2)
    return img


def test_dataset():

    root_dir = r'/data/ylw/dataset/coco'
    set_name='val2017'
    input_shape= [640, 640]
    num_classes=1
    mosaic= True
    mosaic_prob= 0.05
    mixup= True
    mixup_prob= 0.5
    train = False
    ex_kpts=17
    batch_size = 1
    
    coco = COCO(os.path.join(root_dir, 'annotations', f'person_keypoints_{set_name}.json'))
    # coco = COCO(os.path.join(root_dir, 'annotations', f'instances_{set_name}.json'))
    # img_ids = coco.getImgIds()
    # print(f'{len(img_ids)}')
    # cat_ids = coco.getCatIds()
    # print(f'{cat_ids=}')
    # cats = coco.loadCats(cat_ids[0])
    # print(f'{cats=}')
    # for img_id in img_ids:
    #     ann_id = coco.getAnnIds(imgIds=img_id)
    #     # print(f'{ann_id=}')
    #     ann = coco.loadAnns(ann_id)
    #     print(f'{ann=}')
    #     if len(ann) == 0:
    #         print(f'{ann_pid=}  {img_id=}  {ann}')
    #         img_info = coco.loadImgs(img_id)
    #         img_file = os.path.join(root_dir, set_name, img_info['file_name'])
    #         img = cv2.imread(img_file)
    #         cv2.imwrite(f'{img_id}.jpg', img)
    #     for idx, a in enumerate(ann):
    #         if len(a['bbox']) == 0:
    #             print(f'[{idx}]  {a}')
    
    ds = COCODetDataset(
        root_dir, 
        input_shape, 
        num_classes, 
        mosaic, mosaic_prob, 
        mixup, mixup_prob, 
        train, 
        set_name=set_name,
        coco=coco,
        special_aug_ratio = 0.7, 
        ex_kpts=ex_kpts,
        class_list=['person']
    )
    ds.check_coco()
    
    # # # for idx, batch in enumerate(next(iter(ds))):
    # # for batch in next(iter(ds)):
    # #     print(f'{type(batch)=}  {batch.shape=}')
    # #     break


    from torch.utils.data import DataLoader
    
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=ds.yolox_dataset_collate)
    
    for batch in dl:
        print(f'{type(batch)}')
        imgs, ts = batch
        print(f'8888888888888888888888888888888888888888888888888888888{imgs.shape=}')
        for t in ts:
            print(f'{t.shape=}')
        
        for i in range(batch_size):
            img = imgs[i, :, :, :].numpy().transpose((1,2,0))*255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(f'{img.shape=}')
            # cv2.imwrite(f'./test/imgs/{i}_ori.jpg', img)
            # print(f'cls: {ts[i][:, 4]}')
            print(f'{ts[i].shape=}  {ts[i][:, :4]}')
            # print(f'bbox: {ts[i][:, 0:4]}')
            # print(f'kpts: {ts[i][:, 5:]}')
            h,w,_ = img.shape
            tt = ts[i]
            bboxes = ts[i][:, 0:4].numpy() # cxcywh
            bboxes[:, 0] -= bboxes[:, 2]/2
            bboxes[:, 1] -= bboxes[:, 3]/2
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]
            img = rectangle(img, bboxes)
            kpts = ts[i][:, 5:].numpy().reshape((-1, 3))
            img = circle(img, kpts)
            cv2.imwrite(f'./test/imgs/{i}_draw.jpg', img)
            print(f'/'*80)
            
        print(f'='*80)
        
        time.sleep(2)
        break


def get_coco():
    root_dir = r'/data/ylw/dataset/coco'
    set_name='val2017'
    coco = COCO(os.path.join(root_dir, 'annotations', f'person_keypoints_{set_name}.json'))
    
    img_id = 296649
    
    ann_id = coco.getAnnIds(imgIds=img_id)
    print(f'{ann_id=}')
    ann = coco.loadAnns(ann_id)
    print(f'{len(ann)=}/{ann=}')
    img_info = coco.loadImgs(img_id)
    print(f'{img_info=}')
    img_file = os.path.join(root_dir, set_name, img_info[0]['file_name'])
    img = cv2.imread(img_file)
    bboxes = []
    kpts = []
    for a in ann:
        if len(a['bbox']) > 0:
            bboxes.append(a['bbox'])
        if a['num_keypoints'] > 0:
            kpts.append(a['keypoints'])
    bboxes = np.array(bboxes)
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    kpts = np.array(kpts).reshape(-1, 3)
    img = rectangle(img, bboxes)
    img = circle(img, kpts)
    
    cv2.imwrite(f'./test/tmp/{img_id}.jpg', img)




if __name__ == '__main__':
    test_dataset()
    # get_coco()