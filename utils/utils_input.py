import os, sys
import cv2
import PIL.Image as Image
import numpy as np
import copy


class InputOPs:
    def norm(inputs, mode=1, mean=None, std=None, copy=False):
        '''
        model: 
            0: /255
            1: (input - 127.5) / 128
            2: (input - mean) / std
        '''
        inputs_ = inputs
        if copy:
            inputs_ = copy.deepcopy(inputs)
            
        if mode == 0:
            inputs_ /= 255.0
        elif mode == 1:
            inputs_ = (inputs_ - 127.5) / 128.0
        elif mode == 2:
            assert isinstance(mean, None) or isinstance(std, None), f'{mean=} or {std=} is None!!!'
            inputs_ = (inputs_ - mean) / std
            
        return inputs_


class ImageIO:
    def shape(img):
        if isinstance(img, Image.Image):
            return np.array(img).shape
        elif isinstance(img, np.ndarray):
            return img.shape
    
    def ndarray2Image(img):
        return Image.fromarray(img)
    
    def Image2ndarray(img):
        return np.array(img)
    
    def read_image(image, mode='rgb', read_io='cv'):
        MODE = {'rgb': cv2.IMREAD_COLOR, 'gray': cv2.IMREAD_GRAYSCALE, 'ori': cv2.IMREAD_UNCHANGED}
        if isinstance(image, str):
            read_ = lambda x: cv2.imread(x, MODE['rgb']) if read_io == 'cv' else Image.open(x)
            convert_color = lambda img, color='rgb': cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if read_io == 'cv' else img.convert('RGB')
            img = convert_color(read_(image), color='rgb')
        elif isinstance(image, np.ndarray):
            img = image
        else:
            img = None
        return img


if __name__ == '__main__':
    
    img = r'/home/yangliwei/code/yolo/yolov5_pl/img/street.jpg'
    img = ImageIO.read_image(img, mode='rgb', read_io='cv')
    print(f'{ImageIO.shape(img)=}')
