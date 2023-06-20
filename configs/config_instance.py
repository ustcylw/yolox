import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_classes
import time
from prettytable import PrettyTable as PTable





class BASECONFIGS():
    ## envs
    MASTER_ADDR = '127.0.0.1'
    MASTER_PORT = '29500'
    NCCL_DEBUG = "INFO"
    WORLD_SIZE = 2
    
    ## exps
    TIME = time.strftime("%Y:%m:%d-%H:%M:%S")
    EXP_NUM = '000002'
    PRE_SYMBOL = 'yolox_coco_instance'
    ROOT_DIR = './'
    EXP_DIR = os.path.join(ROOT_DIR, 'exps', f'{EXP_NUM}_{PRE_SYMBOL}')
    LOGS_DIR = EXP_DIR
    CKPTS_DIR = os.path.join(ROOT_DIR, 'exps', f'{EXP_NUM}_{PRE_SYMBOL}', TIME, 'ckpts')
    
    LOG_TYPE = 'tb'  # wtb[wandb] tb[tensorboard]
    
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)


class TRAINCONFIGS(BASECONFIGS):


    ## train
    start_epochs = 0
    epochs = 300
    unfreeze_epoch      = -1
    train = True
    sync_bn         = False
    precision = 32  # 是否半精度训练

    model_path      = r'./exps/000002_yolox_voc_instance_v3/lightning_logs/version_3_albu/epoch=000117-train-loss=2.8496-val-loss=2.8869.ckpt'  # fintune预训练模型
    input_shape     = [640, 640]
    backbone        = "resnet18"
    phi             = 's'  # 'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x'
    pretrained      = True  # 通用预训练模型
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    mixup_alhpa          = 0.5
    mixup_beta          = 0.5
    special_aug_ratio   = 0.7
    random_scale        = 0.2
    ex_kpts = 0
    ex_layer = 1
    limit_range = [[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    save_period         = 10
    eval_period         = 10

    # ema
    ema                 = True
    ema_steps           = 16
    ema_decay           = 0.99998
    ema_start_epochs    = int(epochs * 0.8 + start_epochs)

    batch_size    = 32
    devices = [1,6,7,9]
    
    # dataset
    # dataset_root = r'/data8022/ylw/dataset/voc2coco'
    # dataset_name = 'voc2coco'
    # dataset_set_name = 'train'
    # classes_path    = 'model_data/voc_classes.txt'
    dataset_root = r'/data8022/ylw/dataset/coco'
    dataset_name = 'coco'
    dataset_set_name = 'train2017'
    num_workers         = 4
    classes_path    = 'model_data/coco_classes.txt'
    if ex_kpts > 0:
        cls_names, num_classes = ['person'], 1
        class_list = ['person']  # None, ['person']
    else:
        cls_names, num_classes = get_classes(classes_path)
        class_list = None  # None, ['person']
    norm_bbox = True

    # monitor
    monitor = 'val-loss'
    monitor_mode = 'min'
    
    # optim
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4

    # lr
    init_lr             = 6e-3
    min_lr              = init_lr * 0.0001
    gamma               = 0.314
    lr_decay_type       = "multistep"  # "cosine"
    lr_step_size        = 3
    lr_milestone  = [10, 30, 80, 150, 240]  # [10, 20, 50, 80]


class PREDCONFIGS(BASECONFIGS):
    
    ## task
    cls_path    = 'model_data/voc_classes.txt'
    cls_names, num_classes = get_classes(cls_path)
    input_shape     = [640, 640]

    # model_path        = './exps/000001_yolox_coco_instance/lightning_logs/version_1/last.ckpt'
    model_path        = './exps/000002_yolox_voc_instance_v3/lightning_logs/version_3/last.ckpt'
    # model_path        = './exps/000002_yolox_voc_instance_v3/lightning_logs/version_0_if/last.ckpt'
    strides         = [8, 16, 32, 64, 128]
    ex_layer = 1
    ex_kpts = 0
    limit_range = [[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #--------------------------------------------------------------------------#
    #   用于选择所使用的模型的主干
    #   resnet50, hourglass
    #--------------------------------------------------------------------------#
    backbone          = 'resnet18'
    phi             = 's'
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = 'pred_dir'  # pred_image  pred_dir  pred_video
    norm_bbox = True
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "./data/imgs/"
    dir_save_path   = "./exps/runs/pred_dir"
    #--------------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #--------------------------------------------------------------------------#
    confidence        = 0.5
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    nms_iou           = 0.3
    #--------------------------------------------------------------------------#
    #   是否进行非极大抑制，可以根据检测效果自行选择
    #   backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
    #--------------------------------------------------------------------------#
    nms               = False
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    letterbox_image   = False
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    cuda              = True
    device_id = 9



        
    def to_string():
        return ''



class EVALCONFIGS(BASECONFIGS):
    
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'coco_map'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    mode = 'coco_map'
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    ## eval
    confidence = 0.02
    nms_iou = 0.5
    train = False
    # MINOVERLAP      = 0.5
    
    ## model
    backbone        = 'resnet50'
    output_scale=4
    model_path = r'/data/ylw/code/yolo/yolox-pytorch/yolox-v3/exps/000002_yolox_voc_instance_v3/lightning_logs/version_3/last.ckpt'
    strides         = [8, 16, 32, 64, 128]
    special_aug_ratio   = 0.7
    ex_kpts = 0
    ex_layer = 1
    limit_range = [[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    
    batch_size    = 24
    num_workers         = 4

    letterbox_image = False
    cuda = True
    device_id = 0
    
    # dataset
    input_shape     = [640, 640]
    # dataset_root = r'/data8022/ylw/dataset/voc2coco'
    # dataset_root = r'/data/ylw/dataset/voc2coco'
    # dataset_name = 'voc2coco'
    # dataset_set_name = 'val'
    # classes_path    = 'model_data/voc_classes.txt'
    dataset_root = r'/data8022/ylw/dataset/coco'
    dataset_name = 'coco'
    dataset_set_name = 'val2017'
    classes_path    = 'model_data/coco_classes.txt'
    if ex_kpts > 0:
        cls_names, num_classes = ['person'], 1
        class_list = ['person']  # None, ['person']
    else:
        cls_names, num_classes = get_classes(classes_path)
        class_list = None  # None, ['person']
    norm_bbox = True

    input_shape     = [640, 640]
    backbone        = "resnet18"
    phi             = 's'
    pretrained      = True  # 通用预训练模型
    mosaic              = False
    mosaic_prob         = 0.5
    mixup_alhpa          = 0.5
    mixup_beta          = 0.5
    mixup               = False
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
