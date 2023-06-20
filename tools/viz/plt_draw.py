import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import time
from math import *
import cv2
import colorsys




# refs:
#     好——https://blog.csdn.net/sinat_22510827/article/details/90693385
#     http://www.coolpython.net/data_analysis/matplotlib/matplotlib-basic-line-color.html
#     https://blog.csdn.net/claroja/article/details/71123012
#     https://www.osgeo.cn/matplotlib/api/_as_gen/matplotlib.lines.Line2D.html
#     https://blog.csdn.net/stefanjoe/article/details/112107298
#     http://c.biancheng.net/matplotlib/axes-limits.html
#     https://blog.csdn.net/u010021014/article/details/110393223
#     https://vimsky.com/examples/usage/matplotlib-axes-axes-draw-in-python.html
    
    

class PLTDraw():
    def __init__(self):
        self.fig = None
        self.axes = {}
        self.artists = {}
        self.last_line_data = {}
        
        hsv_tuples = [(x / 255, 1., 1.) for x in range(255)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def subplot(self, nrows=1, ncols=1, axes_names=None):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
        self.fig = fig
        ax_names = {axes_names[i][j]: axes[i][j] for i in range(nrows) for j in range(ncols)} if isinstance(axes, list) else {axes_names: axes}
        self.axes.update(ax_names)
    
    def draw_image(
        self, 
        data, 
        axes_name='default-axes', 
        artist_name='default-artist', 
        show=True,
        keep=True
    ):
        '''
        marker: 'o', '^'
        color: 'r', 'b', 'g'
        '''
        axes = self.axes[axes_name]
        # if not keep:
        #     axes.clear()

        # axes.plot(t, m, mode)
        if artist_name in self.artists:
            artist = self.artists[artist_name]
            # artist.set_data(data)
            artist.set_array(data)
        else:
            artist = axes.imshow(data)
            self.artists[artist_name] = artist

        if show:
            renderer = self.fig.canvas.get_renderer()
            axes.draw(renderer)
        
    def draw_line(
        self, 
        data, 
        axes_name='default-axes', 
        artist_name='default-artist', 
        show=True,
        keep=True,
        marker='-',
        color='r'
    ):
        '''
        marker: 'o', '^'
        color: 'r', 'b', 'g'
        '''
        axes = self.axes[axes_name]
        if keep:
            last_data = self.last_line_data.get(artist_name, data)
            X, Y = [last_data[0], data[0]], [last_data[1], data[1]]
        else:
            X, Y = data[0], data[1]
            # axes.clear()

        # axes.plot(t, m, mode)
        if artist_name in self.artists:
            artist = self.artists[artist_name]
            artist.set_data(X, Y)
            print(f'{X=}  {Y=}  {self.last_line_data=}  {data=}')

            if keep:
                self.last_line_data[artist_name] = data
        else:
            artist = Line2D([0], [0])
            artist.set_figure(self.fig)
            # artist = axes.plot(data)[0]
            axes.add_line(artist)
            self.artists[artist_name] = artist
        artist.set_marker(marker)
        artist.set_color(color)
        
        
        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()
        axes.set_xlim(min(data[0].min(), xmin), max(data[0].max(), xmax))
        axes.set_ylim(min(data[1].min(), ymin), max(data[1].max(), ymax))

        if show:
            renderer = self.fig.canvas.get_renderer()
            axes.draw(renderer)

    def draw_bbox(
        self, 
        data, 
        axes_name='default-axes', 
        artist_name='default-artist', 
        show=True,
        keep=True,
        marker='-',
        color='r'
    ):
        '''
        marker: 'o', '^'
        color: 'r', 'b', 'g'
        '''
        axes = self.axes[axes_name]
        if keep:
            last_data = self.last_line_data.get(artist_name, data)
            X, Y = [last_data[0], data[0]], [last_data[1], data[1]]
        else:
            X, Y = data[0], data[1]
            # axes.clear()

        # axes.plot(t, m, mode)
        if artist_name in self.artists:
            artist = self.artists[artist_name]
            artist.set_data(X, Y)
            print(f'{X=}  {Y=}  {self.last_line_data=}  {data=}')

            if keep:
                self.last_line_data[artist_name] = data
        else:
            artist = Line2D([0], [0])
            artist.set_figure(self.fig)
            # artist = axes.plot(data)[0]
            axes.add_line(artist)
            self.artists[artist_name] = artist
        artist.set_marker(marker)
        artist.set_color(color)
        
        
        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()
        axes.set_xlim(min(data[0].min(), xmin), max(data[0].max(), xmax))
        axes.set_ylim(min(data[1].min(), ymin), max(data[1].max(), ymax))

        if show:
            renderer = self.fig.canvas.get_renderer()
            axes.draw(renderer)


def test_1():
    '''
    需要保存历史数据
    '''
    plt.ion() #开启interactive mode 成功的关键函数
    t = [0]
    t_now = 0
    m = [sin(t_now)]

    fig, ax = plt.subplots()

    for i in range(20):
        # plt.clf() #清空画布上的所有内容
        ax.clear()
        t_now = i*0.1
        t.append(t_now)#模拟数据增量流入，保存历史数据
        m.append(sin(t_now))#模拟数据增量流入，保存历史数据
        ax.plot(t, m, '-r')
        # renderer = fig.canvas.get_renderer()
        # ax.draw(renderer)
        fig.canvas.draw()  # ax.draw()都可以
        # time.sleep(0.01)
        plt.pause(0.1)

    plt.ioff()


def test_2():
    '''
    无需保存数据
    '''
    plt.ion() #开启interactive mode 成功的关键函数
    fig, ax = plt.subplots()

    for i in range(20):
        # plt.clf() #清空画布上的所有内容
        # ax.clear()  # 畫布內容保留
        t_now = i*0.1
        ax.plot(t_now, sin(t_now), '.-')
        # renderer = fig.canvas.get_renderer()
        # ax.draw(renderer)
        fig.canvas.draw()  # ax.draw()都可以
        # time.sleep(0.01)
        plt.pause(0.1)

    plt.ioff()
    

def test_3():
    '''
    ???
    无需保存数据（进阶版）
    以上是动态的显示一个函数，也即直观上一条轨迹不断的延伸。这是一种应用，另一种应用是在一张画布上增量式的画多条轨迹（函数）。
    '''
    plt.ion() #开启interactive mode 成功的关键函数
    plt.figure(1)
    t = np.linspace(0, 20, 100)

    for i in range(1):
        # plt.clf() # 清空画布上的所有内容。此处不能调用此函数，不然之前画出的轨迹，将会被清空。
        y = np.sin(t*(i+1)/10.0)
        print(f'{t=}  \n{y=}')
        plt.plot(t, y) # 一条轨迹
        plt.draw()#注意此函数需要调用
        time.sleep(10)
    
    plt.ioff()



def test_4():
    '''
    image
    '''
    plt.ion() #开启interactive mode 成功的关键函数
    fig, ax = plt.subplots()

    img_list = [
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000000.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000001.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000002.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000003.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000004.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000005.png'
    ]
    for i in range(10):
        img = cv2.imread(img_list[i%len(img_list)])
        # plt.clf() #清空画布上的所有内容
        ax.clear()  # 使用Artist，千萬不要使用axes進行clear
        ax.imshow(img)
        renderer = fig.canvas.get_renderer()
        ax.draw(renderer)
        # fig.canvas.draw()  # ax.draw()都可以

        plt.pause(1)

    plt.ioff()
    

def test_4_1():
    '''
    image-artist
    '''
    plt.ion() #开启interactive mode 成功的关键函数
    fig, ax = plt.subplots()

    img_list = [
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000000.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000001.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000002.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000003.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000004.png',
        '/data/ylw/datasets/kitti/data/data_object_image_2/training/image_2/000005.png'
    ]
    img_axes = None
    for i in range(10):
        img = cv2.imread(img_list[i%len(img_list)])

        # plt.clf() #清空画布上的所有内容
        # ax.clear()  # 使用Artist，千萬不要使用axes進行clear
        if img_axes is not None:
            # img_axes.set_data(img)
            img_axes.set_array(img)
        else:
            img_axes = ax.imshow(img)
        renderer = fig.canvas.get_renderer()
        ax.draw(renderer)
        # fig.canvas.draw()  # ax.draw()都可以
        # img_axes.draw(renderer)

        plt.pause(1)

    plt.ioff()
 



if __name__ == '__main__':
    
    if False:
        plt.ion() #开启interactive mode 成功的关键函数
        t = [0]
        t_now = 0
        m1 = [sin(t_now)]
        m2 = [cos(t_now)]
        
        draw = PLTDraw()
        axes_name = 'line1'
        artist_name1 = 'line11'
        artist_name2 = 'line12'
        draw.subplot(1, 1, axes_name)
        mode = '-g'
        keep = False

        for i in range(20):
            
            t_now = i*0.1
            if keep:
                t = [t_now]
                m1 = [sin(t_now)]#模拟数据增量流入，保存历史数据
                m2 = [cos(t_now)]#模拟数据增量流入，保存历史数据
            else:
                t.append(t_now)#模拟数据增量流入，保存历史数据
                m1.append(sin(t_now))#模拟数据增量流入，保存历史数据
                m2.append(cos(t_now))#模拟数据增量流入，保存历史数据
            print(f'[]  {m1=}  {m2=}')
            
            draw.draw_line((np.array(t),np.array(m1)), axes_name, artist_name1, show=True, keep=keep, marker=None, color='g')
            draw.draw_line((np.array(t),np.array(m2)), axes_name, artist_name2, show=True, keep=keep, marker=None, color='b')
            
            plt.pause(0.05)

        plt.ioff()


    if True:

        plt.ion() #开启interactive mode 成功的关键函数
        img_list = [
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/001.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/002.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/003.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/004.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/005.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/006.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/007.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/008.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/009.jpg',
            '/data8022/ylw/code/yolo/yolox-pytorch/src_test_1/img/010.jpg',
        ]

        draw = PLTDraw()
        axes_name = 'image'
        artist_name = 'images'
        draw.subplot(1, 1, axes_name)

        for i in range(10):
            img = cv2.imread(img_list[i%len(img_list)])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            draw.draw_image(img, axes_name, artist_name, show=True, keep=False)

            plt.pause(0.5)

        plt.ioff()
