import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import cv2
import math
import colorsys






# refs:
#   https://blog.51cto.com/u_15088375/5845886


# def bytes2img(img_file):
#     with open(img_file, "wb") as f:
#         f.write(file)
#     img = Image.open(BytesIO(file))
#     img.show()


def img2bytes(img):
    image = img
    if isinstance(img, str):
        image = Image.open(img)
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    return img_bytes


def img2cv(img):
    image = img
    if isinstance(img, str):
        image = Image.open(img)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def cv2img(img):
    image = img
    if isinstance(img, str):
        img = cv2.imread(img)
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


def new(mode='RGB', size=(640, 640), color='gray'):
    return Image.new(mode, size, color=color)


def resize(img, scale=1.0, new_size=None, resample=Image.BILINEAR, copy=False):
    '''
        resample:
            Image.NEAREST ：低质量
            Image.BILINEAR: 双线性
            Image.BICUBIC ：三次样条插值
            Image.ANTIALIAS: 高质量
    '''
    img_size = img.size
    if scale > 0 and scale != 1.0:
        new_size_ = (int(img_size[0]*scale), int(img_size[1]*scale))
    elif new_size is not None:
        new_size_ = new_size
    new_img = img
    if copy:
        new_img = img.copy()
    
    return new_img.resize(new_size_, resample=resample)


def set_size(img, limit_size=[640, 640], resample=Image.BILINEAR, copy=False):
    img_size = img.size
    scale = max(np.asarray(img_size)/np.array(limit_size))
    new_size = np.asarray(img_size) / scale

    img_ = img
    if copy:
        img_ = img_.copy()
    
    img_ = img_.resize((int(new_size[0]), int(new_size[1])))

    new_img = Image.new('RGB', limit_size, color=(128, 128, 128))
    new_img = paste(new_img, img_, center=True)
    return new_img


def paste(img1, img2, lt=(0, 0), center=False, copy=False):
    img1_ = img1
    if copy:
        img1_ = img1.copy()
    lt_ = lt
    if center:
        w1, h1 = img1_.size
        w2, h2 = img2.size
        lt_ = (int((w1-w2)/2), int((h1-h2)/2))
    img1_.paste(img2, lt_)
    return img1_



################################################################################
# bbox
################################################################################
def check_bbox(bboxes, shape):
    '''
        bboxes: np.array([[xyxy], [xyxy]])
        shape: (w, h)
    '''
    if bboxes[:, (0, 1)] < 0:
        bboxes[:, (0, 1)] = 0
    if bboxes[:, 2] > shape[0]:
        bboxes[:, 2] = shape[0]
    if bboxes[:, 3] > shape[1]:
        bboxes[:, 3] = shape[1]


def xyxy2xywh(bboxes, copy=False):
    bboxes_ = bboxes
    if copy:
        bboxes_ = bboxes.copy()
    bboxes_[:, 2] -= bboxes_[:, 0]
    bboxes_[:, 3] -= bboxes_[:, 1]
    return bboxes_


def xywh2xyxy(bboxes, copy=False):
    bboxes_ = bboxes
    if copy:
        bboxes_ = bboxes.copy()
    bboxes_[:, 2] += bboxes_[:, 0]
    bboxes_[:, 3] += bboxes_[:, 1]
    return bboxes_


def xyxy2cxywh(bboxes, copy=False):
    bboxes_ = bboxes
    if copy:
        bboxes_ = bboxes.copy()
    bboxes_[:, 2] -= bboxes_[:, 0]
    bboxes_[:, 3] -= bboxes_[:, 1]
    bboxes_[:, 0] += bboxes_[:, 2]/2
    bboxes_[:, 1] += bboxes_[:, 3]/2
    return bboxes_


def cxywh2xyxy(bboxes, copy=False):
    bboxes_ = bboxes
    if copy:
        bboxes_ = bboxes.copy()
    bboxes_[:, 0] -= bboxes_[:, 2]/2
    bboxes_[:, 1] -= bboxes_[:, 3]/2
    bboxes_[:, 2] += bboxes_[:, 0]
    bboxes_[:, 3] += bboxes_[:, 1]
    return bboxes_


def img2cv(img: Image, mode='rgb'):
    return np.array(img)


def cv2img(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)




################################################################################
# draw
################################################################################

class PILDraw():
    def __init__(self, color_num=80, default_size=640, font_file=r'./tools/viz/simhei.ttf') -> None:
        self.font_file = font_file
        self.default_size = default_size
        hsv_tuples = [(x / color_num, 1., 1.) for x in range(color_num)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def rectangle(self, img, bboxes, labels=None, fille=None, outline=None):
        font = ImageFont.truetype(font=self.font_file, size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
        thickness   = int(max((img.size[0] + img.size[1]) // self.default_size, 1))
        draw_img = ImageDraw.Draw(img)
        
        # for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
        #     c = label['cls']
        for idx, bbox in enumerate(bboxes):
            c = 0
            label = ''
            if labels is not None:
                label = labels[idx]
                c = label['cls']
                label = ' '.join([f'{k}:{v if not isinstance(v, float) else np.round(v, 2)}' for k,v in label.items()])
            left, top, right, bottom = bbox
            label_size = draw_img.textsize(label, font)
            label = label.encode('utf-8')
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for idx1 in range(thickness):
                # draw_img.rectangle([left, top, right, bottom], outline=self.colors[c])
                draw_img.rectangle([left + idx1, top + idx1, right - idx1, bottom - idx1], outline=self.colors[c])
            
            draw_img.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw_img.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            
        del draw_img
        
        return img

    def points(self, img, points, num_points=17, labels=None):
        '''
        points: <n, 34/51>
        labels: 
        '''
        font = ImageFont.truetype(font=self.font_file, size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
        thickness   = int(max((img.size[0] + img.size[1]) // self.default_size, 1))
        draw_img = ImageDraw.Draw(img)
        
        print(f'{points.shape=}')
        for idx, point in enumerate(points):
            point = point.reshape((-1, 3)) if point.shape[0] == num_points*3 else point.reshape((-1, 2))
            point = point.astype(np.int32)
            for p in point:
                # draw_img.point(p[0:2], fill=self.colors[idx])
                draw_img.ellipse((p[0]-thickness, p[1]-thickness, p[0]+thickness, p[1]+thickness), fill=self.colors[idx])
        
        del draw_img
        
        return img






class CVDraw():
    
    def __init__(self, img=None, color_num=80) -> None:
        self.img = img
        self.font_face = cv2.FONT_ITALIC
        
        hsv_tuples = [(x / color_num, 1., 1.) for x in range(color_num)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def optimal_font_dims(self, img, font_scale = 1e-3, thickness_scale = 1e-3):
        h, w, _ = img.shape
        font_scale = np.floor(font_scale * max(w, h)).astype('int32')
        font_scale = 1 if font_scale == 0 else font_scale
        thickness = max(math.ceil((w+h)*thickness_scale), 1)
        return font_scale, thickness

    def rectangle(self, img, bboxes, labels=None, center=False, copy=False, corner=False):
        '''
        bboxes: xyxy
        '''
        img_ = img
        if img is None:
            img_ = self.img
        if copy:
            img_ = img.copy()
        
        if labels is not None and bboxes.shape[0] != len(labels):
            print(f'bboxes length[{bboxes.shape[0]}] do not equals to labels length[{len(labels)}] !!!')
        
        font_scale, thickness = self.optimal_font_dims(img_)
        
        color_ = [(0, 0, 255) for i in range(bboxes.shape[0])]
        if labels is not None:
            color_ = [self.colors[label['cls']] for label in labels]
            # color_ = [np.random.randint(0, 80) for label in labels]
            
        for idx, bbox in enumerate(bboxes):
            bbox = bbox.astype(np.int32)
            img_ = cv2.rectangle(
                img_, 
                (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                color=color_[idx], 
                thickness=thickness
            )
            if center:
                img_ = cv2.circle(img_, (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)), radius=thickness, color=color_[idx], thickness=thickness*2)
            if corner:
                img_ = cv2.circle(img_, (int(bbox[0]), int(bbox[1])), radius=thickness, color=color_[idx], thickness=thickness*2)
                img_ = cv2.circle(img_, (int(bbox[2]), int(bbox[1])), radius=thickness, color=color_[idx], thickness=thickness*2)
                img_ = cv2.circle(img_, (int(bbox[2]), int(bbox[3])), radius=thickness, color=color_[idx], thickness=thickness*2)
                img_ = cv2.circle(img_, (int(bbox[0]), int(bbox[3])), radius=thickness, color=color_[idx], thickness=thickness*2)
            if labels is not None:
                label = labels[idx]#.encode('utf-8')
                # text = ' '.join([f'{k}:{v:2f}' if isinstance(v, float) else f'{k}:{v}' for k, v in label.items()])
                text = ' '.join([f'{k}:{v}' for k, v in label.items()])
                textSize, baseline = cv2.getTextSize(text, self.font_face, font_scale, thickness)
                textSizeWidth, textSizeHeight = textSize
                img_ = cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[0]+textSizeWidth, bbox[1]+textSizeHeight+baseline), color=color_[idx], thickness=-1)
                img_ = cv2.putText(img_, text, (bbox[0], bbox[1]+textSizeHeight+baseline), self.font_face, font_scale, (0,0,0), thickness)
                # img_ = Image.fromarray(img_)
                # draw_img = ImageDraw.Draw(img_)
                # font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * img_.size[1] + 0.5).astype('int32'))
                # label_size = draw_img.textsize(text, font)
                # text = text.encode('utf-8')
                # if bbox[1] - label_size[1] >= 0:
                #     text_origin = np.array([bbox[0], bbox[1] - label_size[1]])
                # else:
                #     text_origin = np.array([bbox[0], bbox[1] + 1])
                # draw_img.text(text_origin, str(text,'UTF-8'), fill=(0, 0, 0), font=font)
                # del draw_img
                # img_ = np.array(img_)
                
        return img_

    def rectangle_cn(self, img, bboxes, labels=None, center=False, copy=False, corner=False):
        img_ = img
        if img is None:
            img_ = self.img
        if copy:
            img_ = img.copy()
        
        if bboxes.shape[0] != len(labels):
            print(f'bboxes length[{bboxes.shape[0]}] do not equals to labels length[{len(labels)}] !!!')
        
        thickness = self.get_thickness(img_.shape, 640)
        color = (255, 0, 0)
        if labels is not None:
            img_text = Image.fromarray(img)
            draw_img = ImageDraw.Draw(img_text)
            
        for idx, bbox in enumerate(bboxes):
            bbox = bbox.astype(np.int32)
            img_ = cv2.rectangle(
                img_, 
                (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                color=color, 
                thickness=thickness
            )
            if center:
                img_ = cv2.circle(img_, (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)), radius=thickness, color=color, thickness=thickness*2)
            if corner:
                img_ = cv2.circle(img_, (int(bbox[0]), int(bbox[1])), radius=thickness, color=color, thickness=thickness*2)
                img_ = cv2.circle(img_, (int(bbox[2]), int(bbox[1])), radius=thickness, color=color, thickness=thickness*2)
                img_ = cv2.circle(img_, (int(bbox[2]), int(bbox[3])), radius=thickness, color=color, thickness=thickness*2)
                img_ = cv2.circle(img_, (int(bbox[0]), int(bbox[3])), radius=thickness, color=color, thickness=thickness*2)
            if labels is not None:
                label = labels[idx].encode('utf-8')
                label_size = draw_img.textsize(label, self.font)
                if bbox[1] - label_size[1] >= 0:
                    text_origin = np.array([bbox[0], bbox[1] - label_size[1]])
                else:
                    text_origin = np.array([bbox[1], bbox[0] + 1])
                draw_img.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0,0,255))
                draw_img.text(text_origin, label, fill=(0, 0, 0), font=self.font)
                
        return img_

    def circle(self, img, points, labels=None, pos=False, copy=False):
        img_ = img
        if img is None:
            img_ = self.img
        if copy:
            img_ = img.copy()
        
        font_scale, thickness = self.optimal_font_dims(img_)
        color_ = [(0, 0, 255) for i in range(points.shape[0])]
        if labels is not None:
            color_ = [self.colors[label['cls']] for label in labels]

        for idx, point in enumerate(points):
            point = point.astype(np.int32)
            label = labels[idx]#.encode('utf-8')
            text = ' '.join([f'{k}:{v}' for k, v in label.items()])
            img_ = cv2.circle(img_, (int(point[0]), int(point[1])), radius=thickness, color=color_[idx], thickness=thickness*2)
            if pos:
                text = f'({point[0]},{point[1]}) {text}'
            img_ = cv2.putText(img_, text, (int(point[0]), int(point[1])), self.font_face, font_scale, color_[idx])
        return img_







def test_paste():
    img_file1 = r'/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/exps/runs/pred_dir/street.jpg'
    img_file2 = r'/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/exps/runs/pred_dir/027.jpg'
    img1 = Image.open(img_file1)
    img2 = Image.open(img_file2)

    img12 = pil_draw.paste(img1, img2, lt=(0,0), center=True)
    img12.save(f'./test_paste.jpg')


def test_CVDraw():
    import os, sys
    sys.path.append(r'/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/viz')
    import pil_draw
    import PIL.Image as Image
    import numpy as np


    font_file = '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/model_data/simhei.ttf'
    img_file = r'/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/exps/runs/pred_dir/street.jpg'
    # img_file = r'/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/exps/runs/pred_dir/001.jpg'
    img = Image.open(img_file)


    bboxes = np.array([[50, 50, 300, 500], [100, 150, 800, 800]])
    img = pil_draw.CVDraw(img).rectangle(
        img, 
        bboxes, 
        labels=[
            {'person': 0.99, 'score': 0.85, 'cls':0}, 
            {'人': 0.99, '得分': 0.85, 'cls':10}
        ], 
        center=True, corner=False
    )
    img = pil_draw.CVDraw(np.array(img)).circle(
        np.array(img), 
        bboxes.reshape((-1, 2)), 
        labels=[
            {'head': 0.99, 'score': 0.85, 'cls':0}, 
            {'hand1': 0.99, '得分': 0.85, 'cls':10}, 
            {'hand2': 0.99, '得分': 0.85, 'cls':10}, 
            {'hand3': 0.99, '得分': 0.85, 'cls':10}
        ], 
        pos=True
    )

    Image.fromarray(img).save(f'./test_cvdraw.jpg')


if __name__ == '__main__':
    
    import os, sys
    sys.path.append(r'/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/viz')
    import pil_draw
    import PIL.Image as Image

    # test_paste()
    test_CVDraw()