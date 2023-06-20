import numpy as np
from PIL import Image
from io import BytesIO
import cv2








################################################################################
# bytes to PIL.Image
################################################################################
def bytes2Image(img_bytes):
    return Image.open(BytesIO(img_bytes))

def Image2bytes(img):
    img_ = img
    if isinstance(img, np.ndarray):
        img_ = Image.fromarray(img_)
    img_bytes = BytesIO()
    img_.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()

def bytes2cv(img_bytes):
    img_buffer_numpy = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_buffer_numpy, 1)

def cv2bytes(img):
    return cv2.imencode('.jpg', img)[1].tobytes()


################################################################################
# Image 2 cv
################################################################################
def Image2cv(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def cv2Image(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



def gray2rgb(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image 


def bgr2rgb(image):
    if isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image 








