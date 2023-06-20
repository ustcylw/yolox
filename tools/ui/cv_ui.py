import cv2
import numpy as np


def draw(image, bbox):
    image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), (0, 0, 255), 2)
    return image

class Button():
    def __init__(self, image, shape=[20,60,50,250], color=(0, 0, 180), ori=(100, 100), name='button', callback=None, output={}) -> None:
        self.shape = shape
        self.color = color
        self.ori = ori
        self.output = output
        self.name = name
        self.callback = callback
        image[self.shape[0]:self.shape[1],self.shape[2]:self.shape[3]] = self.color
        cv2.putText(image, self.name,(100,50),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)

    def update(self, image):
        image[self.shape[0]:self.shape[1],self.shape[2]:self.shape[3]] = self.color
        cv2.putText(image, self.name,(100,50),cv2.FONT_HERSHEY_PLAIN, 2,(0),3)

    def process_click(self, event, x, y, flags, params):
        # check if the click is within the dimensions of the button
        frame = params['frame']
        wind = params['wind']
        frame_ = frame
        
        bbox = None
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > self.shape[0] and y < self.shape[1] and x > self.shape[2] and x < self.shape[3]: 
                bbox = cv2.selectROI(wind, frame_, fromCenter=False, showCrosshair=True)
                print(f'Clicked on Button!  roi: {bbox}')
                # frame_ = self.callback(frame_, bbox)
                self.output.update({'bbox': bbox})

        return bbox


# function that handles the trackbar
def startCapture(val):
    # check if the value of the slider
    if val == 1:
        print('Capture started!')
    else:
        print('Capture stopped!')            



cap = cv2.VideoCapture(r'D:\code\matplot\001.mp4')
ret, image = cap.read()
roi = {}
b = Button(image, ori=(30, 30), callback=draw, output=roi)

while cap.isOpened():
    ret, image = cap.read()
    image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))

    b.update(image=image)
    print(f'{roi=}')
    # create a window and attach a mousecallback and a trackbar
    cv2.namedWindow('Control')
    cv2.setMouseCallback('Control', b.process_click, param={'frame':image, 'wind': 'Control', 'scale': 0.5})
    cv2.createTrackbar("Capture", 'Control', 0,1, startCapture)
    print(f'[]  {roi=}')
    
    if roi.get('bbox') is not None:
        image = draw(image, roi['bbox'])
    
    #show 'control panel'
    cv2.imshow('Control', image)
    
    if cv2.waitKey(100) == ord('q'):
        break


cv2.waitKey(0)
cv2.destroyAllWindows()
