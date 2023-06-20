import argparse
import cv2
import time

# 第一步进行参数设置
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, default='./001.mp4',
                help='path to input video file')
ap.add_argument('-t', '--tracker', type=str,
                default='kcf', help='Opencv object tracker type')

args = vars(ap.parse_args())


# opencv已经实现的追踪算法
# 第二步：构造cv2已有追踪算法的列表
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


# 第三步：实例化追踪器
# 实例化Opencv's multi-object tracker
trackers = cv2.MultiTracker_create()

# 第四步：使用cv2.VideoCapture读取视频
vs = cv2.VideoCapture(args['video'])

while True:
    # 第五步：读入第一张图片
    frame = vs.read()
    frame = frame[1]
    # 到头了就结束
    if frame is None:
        break

    # 第六步：使用cv2.resize对图像进行长宽的放缩操作
    h, w = frame.shape[:2]
    width = 600
    r = width / float(w)
    dim = (width, int(r * h))
    frame = cv2.resize(frame, dim, cv2.INTER_AREA)

    # 第七步：使用trackers.apply获得矩形框
    (success, boxes) = trackers.update(frame)

    # 第八步：循环多组矩形框，进行画图操作
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 第九步：进行图像展示
    cv2.imshow('Frame', frame)

    # 第十步：判断按键，如果是s的话，进行画出新的box
    key = cv2.waitKey(100) & 0xff

    if key == ord('s'):
        # 第十一步：选择一个区域，按s键，并将tracker追踪器，frame和box传入到trackers中
        box = cv2.selectROI('Frame', frame, fromCenter=False,
                            showCrosshair=True)
        tracker = OPENCV_OBJECT_TRACKERS[args['tracker']]()
        trackers.add(tracker, frame, box)

    elif key == 27:
        break


vs.release()
cv2.destroyAllWindows()