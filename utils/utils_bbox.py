import numpy as np
import torch
from torchvision.ops import nms, boxes



class Decode():
    def __init__(self, configs) -> None:
        self.configs = configs
        self.bbox_util = DecodeBox(input_shape=self.configs.input_shape, num_cls=self.configs.num_classes, ex_kpts=self.configs.ex_kpts)
    
    def decode(self, outputs, image):
        '''
            outputs: model predict
            image: ori-image[PIL.Image]
        '''
        image_shape = np.array(np.shape(image)[0:2])
        outputs = self.bbox_util.decode_box(outputs, self.configs.input_shape)
        #---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        #---------------------------------------------------------#
        results = self.bbox_util.non_max_suppression(
            outputs, 
            self.configs.num_classes, 
            self.configs.input_shape, 
            image_shape, 
            self.configs.letterbox_image, 
            conf_thres = self.configs.confidence, 
            nms_thres = self.configs.nms_iou
        )
                                                
        if results[0] is None: 
            return image, None, None

        top_label   = np.array(results[0][:, 6], dtype = 'int32')
        top_conf    = results[0][:, 4] * results[0][:, 5]
        top_boxes   = results[0][:, :4]
        if self.configs.ex_kpts > 0:
            top_kpts = results[0][..., 7:]

        bboxes = []
        labels = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.configs.cls_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = int(max(0, np.floor(top).astype('int32')))
            left    = int(max(0, np.floor(left).astype('int32')))
            bottom  = int(min(image.size[1], np.floor(bottom).astype('int32')))
            right   = int(min(image.size[0], np.floor(right).astype('int32')))
            print(f'{predicted_class}:{c}-{score:04f}: {left} {top} {right} {bottom}')

            bboxes.append([left, top, right-left, bottom-top, c, score])
            labels.append([predicted_class])
        if self.configs.ex_kpts > 0:
            return image, np.array(bboxes), labels, top_kpts
        else:
            return image, np.array(bboxes), labels
            

class DecodeBox():
    def __init__(self, input_shape=[640, 640], num_cls=1, ex_kpts=0):
        super(DecodeBox, self).__init__()
        self.input_shape    = input_shape
        self.num_cls    = num_cls
        self.ex_kpts    = ex_kpts

    def decode_box(self, outputs, input_shape):
        grids   = []
        strides = []
        hw      = [x.shape[-2:] for x in outputs] # hw=[[80, 80], [40, 40], [20, 20]]
        #---------------------------------------------------#
        #   outputs输入前代表每个特征层的预测结果
        #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
        #   batch_size, 5 + num_classes, 40, 40
        #   batch_size, 5 + num_classes, 20, 20
        #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
        #   堆叠后为batch_size, 8400, 5 + num_classes
        #---------------------------------------------------#
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)  # outputs: [1, 8400, 57]
        #---------------------------------------------------#
        #   获得每一个特征点属于每一个种类的概率
        #---------------------------------------------------#
        outputs[:, :, 4:5+self.num_cls] = torch.sigmoid(outputs[:, :, 4:5+self.num_cls])  # outputs[:, :, 4:5+self.num_cls]: [1, 8400, 2]
        
        if self.ex_kpts > 0:
            # 对每个点的confidence做阈值归一化
            outputs[:, :, 5+self.num_cls+2::3] = torch.sigmoid(outputs[:, :, 5+self.num_cls+2::3])  # outputs[:, :, 4:5+self.num_cls]: [1, 8400, 2]

        for h, w in hw:
            #---------------------------#
            #   根据特征层的高宽生成网格点
            #---------------------------#   
            grid_y, grid_x  = torch.meshgrid([torch.arange(h), torch.arange(w)])
            #---------------------------#
            #   1, 6400, 2
            #   1, 1600, 2
            #   1, 400, 2
            #---------------------------#   
            grid            = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
            shape           = grid.shape[:2]

            grids.append(grid)
            strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
        #---------------------------#
        #   将网格点堆叠到一起
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        #
        #   1, 8400, 2
        #---------------------------#
        grids               = torch.cat(grids, dim=1).type(outputs.type())
        strides             = torch.cat(strides, dim=1).type(outputs.type())
        #------------------------#
        #   根据网格点进行解码
        #------------------------#
        outputs[..., :2]    = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4]   = torch.exp(outputs[..., 2:4]) * strides

        #-----------------#
        #   归一化
        #-----------------#
        outputs[..., [0,2]] = outputs[..., [0,2]] / input_shape[1]
        outputs[..., [1,3]] = outputs[..., [1,3]] / input_shape[0]
        if self.ex_kpts > 0:
            # 对kpts的xy进行网格点解码
            outputs[..., 5+self.num_cls::3]    = (outputs[..., 5+self.num_cls::3] + grids[..., 0:1]) * strides
            outputs[..., 5+self.num_cls+1::3]    = (outputs[..., 5+self.num_cls+1::3] + grids[..., 1:2]) * strides
            # 对kpts的xy归一化
            outputs[..., 5+self.num_cls::3] = outputs[..., 5+self.num_cls::3] / input_shape[1]
            outputs[..., 5+self.num_cls+1::3] = outputs[..., 5+self.num_cls+1::3] / input_shape[0]
        
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image, kpts=None):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        if kpts is not None:
            kpts[..., 0::3] *= image_shape[1]
            kpts[..., 1::3] *= image_shape[0]
        return boxes if kpts is None else np.concatenate([boxes, kpts], axis=1)

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        
        output = [None for _ in range(len(prediction))]
        #----------------------------------------------------------#
        #   对输入图片进行循环，一般只会进行一次
        #----------------------------------------------------------#
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            #----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            
            if not image_pred.size(0):
                continue
            #-------------------------------------------------------------------------#
            #   detections  [num_anchors, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            #   | 0 | 1 | 2 | 3  |   4      |   5      |   6      |  7    |  8    |   9      | 10    | 11    |   12     |
            #   | x | y | x | y  | obj_conf | cls_conf | cls_pred | kpt-x | kpt-y | kpt-conf | kpt-x | kpt-y | kpt-conf | ...
            #-------------------------------------------------------------------------#
            if self.ex_kpts > 0:
                detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5+self.num_cls:]), 1)
            else:
                detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            
            nms_out_index = boxes.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thres,
            )

            output[i]   = detections[nms_out_index]
            # print(f'[non_max_suppression]  {output[i].shape=}  {output[i][0, 7:]=}  {output[i][1, 7:]=}')

            # #------------------------------------------#
            # #   获得预测结果中包含的所有种类
            # #------------------------------------------#
            # unique_labels = detections[:, -1].cpu().unique()

            # if prediction.is_cuda:
            #     unique_labels = unique_labels.cuda()
            #     detections = detections.cuda()

            # for c in unique_labels:
            #     #------------------------------------------#
            #     #   获得某一类得分筛选后全部的预测结果
            #     #------------------------------------------#
            #     detections_class = detections[detections[:, -1] == c]

            #     #------------------------------------------#
            #     #   使用官方自带的非极大抑制会速度更快一些！
            #     #------------------------------------------#
            #     keep = nms(
            #         detections_class[:, :4],
            #         detections_class[:, 4] * detections_class[:, 5],
            #         nms_thres
            #     )
            #     max_detections = detections_class[keep]
                
            #     # # 按照存在物体的置信度排序
            #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            #     # detections_class = detections_class[conf_sort_index]
            #     # # 进行非极大抑制
            #     # max_detections = []
            #     # while detections_class.size(0):
            #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     #     max_detections.append(detections_class[0].unsqueeze(0))
            #     #     if len(detections_class) == 1:
            #     #         break
            #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     #     detections_class = detections_class[1:][ious < nms_thres]
            #     # # 堆叠
            #     # max_detections = torch.cat(max_detections).data
                
            #     # Add max detections to outputs
            #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                kpts = None
                if self.ex_kpts > 0:
                    kpts = output[i][..., 5+self.num_cls*2:]
                correct_boxes  = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image, kpts)
                output[i][:, :4]  = correct_boxes[..., :4]
                # x1, y1, x2, y2, obj_conf, class_conf, class_pred
                # 5+2*self.num_cls:
                output[i][:, 5+self.num_cls*2:]  = correct_boxes[..., 4:]
        return output
        

class EncodeBox():
    def __init__(self, num_classes, anchors, anchors_mask, input_shape) -> None:
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.bbox_attrs         = 5 + num_classes
        self.threshold          = 4
        self.input_shape          = input_shape
    
    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def encoder(self, targets):
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)  # 3
        
        input_shape = np.array(self.input_shape, dtype='int32')  # [640, 640]
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]  # [array([20, 20], dtype=int32), array([40, 40], dtype=int32), array([80, 80], dtype=int32)]
        y_true      = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]  # (3, 20, 20, 25) (3, 40, 40, 25) (3, 80, 80, 25)
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]  # (3, 20, 20) (3, 40, 40) (3, 80, 80)
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            ## 第l层
            in_h, in_w      = grid_shapes[l]  # 0: [20, 20]  1: [40, 40]  2: [80, 80]
            anchors         = np.array(self.anchors) / {0:32, 1:16, 2:8, 3:4}[l]  # (9, 2)
            
            batch_target = np.zeros_like(targets)
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            batch_target[:, [0,2]]  = targets[:, [0,2]] * in_w  # target(box)归一化到[0,1]区间，在反向放大到每一层
            batch_target[:, [1,3]]  = targets[:, [1,3]] * in_h
            batch_target[:, 4]      = targets[:, 4]
            #-------------------------------------------------------#
            #   wh                          : num_true_box, 2
            #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
            #   anchors                     : 9, 2
            #   np.expand_dims(anchors, 0)  : 1, 9, 2
            #   
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : num_true_box, 9
            #-------------------------------------------------------#
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)  # (num_true_box, 9, 2) = (num_true_box, 1, 2) / (1, 9, 2)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)  # (num_true_box, 9, 2) = (num_true_box, 1, 2) / (1, 9, 2)
            ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)  # (num_true_box, 9, 4)
            max_ratios           = np.max(ratios, axis = -1)  # (num_true_box, 9)
            
            for t, ratio in enumerate(max_ratios):
                # 为每一个box匹配层
                #-------------------------------------------------------#
                #   ratio : 9
                #-------------------------------------------------------#
                over_threshold = ratio < self.threshold  # 比率小于4
                over_threshold[np.argmin(ratio)] = True  # 最小的比率位置，之所以添加这个，是因为可能所有比率都大于4，以至于当前box没法预测
                for k, mask in enumerate(self.anchors_mask[l]):
                    # 选取第l层的mask: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                    if not over_threshold[mask]:
                        # 如果当前层所有ratio都大于4，则进行下一层匹配
                        # 如果当前层存在ratio小于4，则进行当前层匹配
                        continue
                    #----------------------------------------#
                    #   获得真实框属于哪个网格点
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #----------------------------------------#
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))
                    
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            # 如果当前层当前位置，已有box占用
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                # 如果当前占用的box的ratio大于当前box的ratio，则删除当前box位置内容，后边把当前box信息填写进入当前位置。
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                            
                        #----------------------------------------#
                        #   取出真实框的种类
                        #----------------------------------------#
                        c = int(batch_target[t, 4])

                        #----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        #----------------------------------------#
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        #----------------------------------------#
                        #   获得当前先验框最好的比例
                        #----------------------------------------#
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
                        
        return y_true
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    #---------------------------------------------------#
    #   将预测值的每个特征层调成真实值
    #---------------------------------------------------#
    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
        #-----------------------------------------------#
        #   input   batch_size, 3 * (4 + 1 + num_classes), 20, 20
        #-----------------------------------------------#
        batch_size      = input.size(0)
        input_height    = input.size(2)
        input_width     = input.size(3)

        #-----------------------------------------------#
        #   输入为640x640时 input_shape = [640, 640]  input_height = 20, input_width = 20
        #   640 / 20 = 32
        #   stride_h = stride_w = 32
        #-----------------------------------------------#
        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #   anchor_width, anchor_height / stride_h, stride_w
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors[anchors_mask[2]]]

        #-----------------------------------------------#
        #   batch_size, 3 * (4 + 1 + num_classes), 20, 20 => 
        #   batch_size, 3, 5 + num_classes, 20, 20  => 
        #   batch_size, 3, 20, 20, 4 + 1 + num_classes
        #-----------------------------------------------#
        prediction = input.view(batch_size, len(anchors_mask[2]),
                                num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        #-----------------------------------------------#
        #   先验框的中心位置的调整参数
        #-----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        #   先验框的宽高调整参数
        #-----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2]) 
        h = torch.sigmoid(prediction[..., 3]) 
        #-----------------------------------------------#
        #   获得置信度，是否有物体 0 - 1
        #-----------------------------------------------#
        conf        = torch.sigmoid(prediction[..., 4])
        #-----------------------------------------------#
        #   种类置信度 0 - 1
        #-----------------------------------------------#
        pred_cls    = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角 
        #   batch_size,3,20,20
        #   range(20)
        #   [
        #       [0, 1, 2, 3 ……, 19], 
        #       [0, 1, 2, 3 ……, 19], 
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ] * (batch_size * 3)
        #   [batch_size, 3, 20, 20]
        #   
        #   [
        #       [0, 1, 2, 3 ……, 19], 
        #       [0, 1, 2, 3 ……, 19], 
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ].T * (batch_size * 3)
        #   [batch_size, 3, 20, 20]
        #----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)

        #----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        #----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #   x  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_x
        #   y  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_y
        #   w  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_w
        #   h  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_h 
        #----------------------------------------------------------#
        pred_boxes          = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0]  = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1]  = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2]  = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3]  = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5
        
        box_xy          = pred_boxes[..., 0:2].cpu().numpy() * 32
        box_wh          = pred_boxes[..., 2:4].cpu().numpy() * 32
        grid_x          = grid_x.cpu().numpy() * 32
        grid_y          = grid_y.cpu().numpy() * 32
        anchor_w        = anchor_w.cpu().numpy() * 32
        anchor_h        = anchor_h.cpu().numpy() * 32
        
        fig = plt.figure()
        ax  = fig.add_subplot(121)
        from PIL import Image
        img = Image.open("img/street.jpg").resize([640, 640])
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.gca().invert_yaxis()

        anchor_left = grid_x - anchor_w / 2
        anchor_top  = grid_y - anchor_h / 2
        
        rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w],anchor_top[0, 0, point_h, point_w]], \
            anchor_w[0, 0, point_h, point_w],anchor_h[0, 0, point_h, point_w],color="r",fill=False)
        rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w],anchor_top[0, 1, point_h, point_w]], \
            anchor_w[0, 1, point_h, point_w],anchor_h[0, 1, point_h, point_w],color="r",fill=False)
        rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w],anchor_top[0, 2, point_h, point_w]], \
            anchor_w[0, 2, point_h, point_w],anchor_h[0, 2, point_h, point_w],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax  = fig.add_subplot(122)
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.scatter(box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c='r')
        plt.gca().invert_yaxis()

        pre_left    = box_xy[...,0] - box_wh[...,0] / 2
        pre_top     = box_xy[...,1] - box_wh[...,1] / 2

        rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]],\
            box_wh[0, 0, point_h, point_w,0], box_wh[0, 0, point_h, point_w,1],color="r",fill=False)
        rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]],\
            box_wh[0, 1, point_h, point_w,0], box_wh[0, 1, point_h, point_w,1],color="r",fill=False)
        rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]],\
            box_wh[0, 2, point_h, point_w,0], box_wh[0, 2, point_h, point_w,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
        #
    feat            = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
    anchors         = np.array([[116, 90], [156, 198], [373, 326], [30,61], [62,45], [59,119], [10,13], [16,30], [33,23]])
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)
