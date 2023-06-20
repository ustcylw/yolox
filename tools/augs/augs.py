import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from PIL import Image 
from image import image as TImage 
from math import fabs, cos, sin, radians






def rotate_keep(degree, scale, width, height):
    """
    无损图像的旋转
    @param degree:
    @param width:
    @param height:
    @return: 旋转矩阵，新的宽高
    """
    height_new = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    width_new = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, scale)
    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2
    return mat_rotation, width_new, height_new


def rotate_center_keep(img, degree, scale, bboxes=None, kpts=None):
    '''
    img: opencv type
    '''
    rows, cols, channel = img.shape
    M, width_new, height_new = rotate_keep(degree, scale, cols, rows)
    rotated = cv2.warpAffine(img, M, (width_new, height_new))
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
        kpts_ = kpts.copy()
        kpts_mode = kpts_.shape[1]
        kpts__ = kpts_[:, 0:2] if kpts_mode == 3 else kpts_
        kpts__ = np.column_stack((kpts__, np.ones(kpts_.shape[0]).T))
        kpts__ = np.transpose(kpts__, (1, 0))
        kpts__ = np.dot(M, kpts__)
        kpts__ = np.transpose(kpts__, (1, 0))
        kpts_[:, 0:2] = kpts__
        kpts_ = kpts__
    return rotated, bboxes_, kpts_


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
        kpts_ = kpts.copy()
        kpts_mode = kpts.shape[1]
        kpts__ = kpts_[:, 0:2] if kpts_mode == 3 else kpts_
        kpts__ = np.column_stack((kpts__, np.ones(kpts_.shape[0]).T))
        kpts__ = np.transpose(kpts__, (1, 0))
        kpts__ = np.dot(M, kpts__)
        kpts__ = np.transpose(kpts__, (1, 0))
        kpts_[:, 0:2] = kpts__
        kpts_ = kpts__
    return rotated, bboxes_, kpts_


def get_shape(shape1, shape2):
    w,h = shape1
    wi, hi = shape2
    if w/wi > h/hi:
        scale = h/hi
    else:
        scale = w/wi
    
    nw, nh = wi * scale, hi * scale

    return nw, nh, scale

def get_pad(shape1, shape2, shift=True):
    l, t, r, b = 0, 0, 0, 0
    l = r = int((shape1[0] - shape2[0]) /2)
    t = b = int((shape1[1] - shape2[1]) /2)
    return l, t, r, b

def paste(img1, img2, bboxes=None, kpts=None):
    '''
    img: Image
    '''
    new_shape = get_shape(img1.size, img2.size)
    new_img = img2.resize((int(new_shape[0]), int(new_shape[1])))
    l, t, r, b = get_pad(img1.size, new_img.size)
    img1.paste(new_img, (l, t))
    if bboxes is not None:
        bboxes *= new_shape[2]
        bboxes[:, (0, 2)] += l
        bboxes[:, (1, 3)] += t
    if kpts is not None:
        kpts *= new_shape[2]
        kpts[:, 0] += l
        kpts[:, 1] += t
    return img1, bboxes, kpts


def resize_image(image, size, letterbox_image=False, bboxes=None, points=None):
    '''
    image: ndarray
    bboxes: xyxycs
    points: xycs
    '''
    image_ = image
    if isinstance(image_, np.ndarray):
        image_ = Image.fromarray(image_)
        
    iw, ih  = image_.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image_   = image_.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image_, ((w-nw)//2, (h-nh)//2))
        if bboxes is not None:
            bboxes[:, (0, 2)] += (w-nw)//2
            bboxes[:, (1, 3)] += (h-nh)//2
        if points is not None:
            points[:, 0] += (w-nw)//2
            points[:, 1] += (h-nh)//2
    else:
        new_image = image_.resize((w, h), Image.Resampling.BICUBIC)
        scale = w/iw, h/ih
        if bboxes is not None:
            bboxes[:, (0, 2)] *= scale[0]
            bboxes[:, (1, 3)] *= scale[1]
        if points is not None:
            points[:, 0] *= scale[0]
            points[:, 1] *= scale[1]
    return new_image, bboxes, points




class Norm():
    def __init__(self):
        ...
    
    def __call__(self, image, norm='255'):
        if norm == '255':
            return image / 255.0
        elif norm == '128':
            return (image - 127.5) / 128
        else:
            return image



class Rotate():
    def __init__(self, degree=0, scale=1.0):
        self.degree = degree
        self.scale = scale
    
    def __call__(self, image, bboxes=None, kpts=None, degree=None, scale=None):
        degree_ = self.degree if degree is None else degree
        scale_ = self.scale if scale is None else scale
        image_, bboxes_, kpts_ = rotate_center(TImage.Image2cv(image), degree_, scale_, bboxes_, kpts_)
        return image_, bboxes_, kpts_
