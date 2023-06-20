import cv2
import subprocess
import time



class RTSPPush():
    def __init__(self, width, height, fps, push_url="rtsp://127.0.0.1:8090/test"):
        '''推流url地址, 指定 用opencv把各种处理后的流(视频帧) 推到 哪里'''
        self.push_url = push_url
        self.width = width
        self.height = height
        self.fps = fps

        # command = [r'D:\Softwares\ffmpeg-5.1-full_build\bin\ffmpeg.exe', # windows要指定ffmpeg地址
        # self.command = ['ffmpeg', # linux不用指定
        #     '-y', '-an',
        #     '-f', 'rawvideo',
        #     '-vcodec','rawvideo',
        #     '-pix_fmt', 'bgr24', #像素格式
        #     '-s', "{}x{}".format(width, height),
        #     '-r', str(fps), # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。 
        #     '-i', '-',
        #     '-c:v', 'libx264',  # 视频编码方式
        #     '-pix_fmt', 'yuv420p',
        #     '-preset', 'ultrafast',
        #     '-f', 'rtsp', #  flv rtsp
        #     '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
        #     push_url] # rtsp rtmp  
        self.command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            # '-tune:v', 'zerolatency',
            '-preset', 'ultrafast',
            #'-f', 'flv', 
            '-f', 'rtsp',
            push_url] # rtsp rtmp  
        self.pipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE)  # , stderr=subprocess.PIPE)
        print(f'[============]  {self.pipe=}')

    def push(self, frame):
        # while True: # True or video_capture.isOpened():
            
            self.pipe.stdin.write(frame.tobytes())
            # pipe.stdin.write(frame.tobytes())
    
    def destory(self):
        self.pipe.terminate()



if __name__ == '__main__':
    
    import os, sys
    
    video_file = r'/data/ylw/dataset/videos/00000264.mp4'
    video_capture = cv2.VideoCapture(video_file)
    if not video_capture.isOpened():
        print(f'video open error !!!')
        sys.exit(0)

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # push_url="rtsp://127.0.0.1:8090/test"
    push_url="rtsp://192.168.0.100:8090/test"
    push = RTSPPush(width=width, height=height, fps=fps, push_url=push_url)
    
    
    push_idx = 0
    while True:
        
        ret, frame = video_capture.read()
        if not ret:
            print(f'read frame error !!!')
            continue
        frame = cv2.putText(frame, f'{push_idx=}', (30, 30), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 0, 255), thickness=1)
        
        push.push(frame=frame)
        
        # if cv2.waitKey(1000/fps) & 0xff == ord('q'):
        #     push.destory()
        #     break
        print(f'{push_idx=}  {ret=}')
        
    push.destory()
    