#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class YOLOLoss(nn.Module):    
    def __init__(self, num_classes, fp16, strides=[8, 16, 32], ex_kpts=0):
        super().__init__()
        self.num_classes        = num_classes
        self.strides            = strides
        self.ex_kpts            = ex_kpts

        self.bcewithlog_loss    = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss           = IOUloss(reduction="none")
        self.grids              = [torch.zeros(1)] * len(strides)
        self.fp16               = fp16

    def forward(self, inputs, labels=None):
        '''
        output格式:
        [x,y,w,h,obj-c,cls,kpts-x,kpts-y,kpts-c]
        labels格式:
        [x,y,w,h,cls,obj,kpts-x,kpts-y,kpts-c]
        '''
        
        outputs             = []
        x_shifts            = []
        y_shifts            = []
        expanded_strides    = []

        #-----------------------------------------------#
        # inputs    [[batch_size, num_classes + 5 + num_kpts*3, 20, 20]
        #            [batch_size, num_classes + 5 + num_kpts*3, 40, 40]
        #            [batch_size, num_classes + 5 + num_kpts*3, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 5 + num_kpts*3]
        #            [batch_size, 1600, num_classes + 5 + num_kpts*3]
        #            [batch_size, 6400, num_classes + 5 + num_kpts*3]]
        # x_shifts  [[batch_size, 400]
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        #-----------------------------------------------#
        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1))

    def get_output_and_grid(self, output, k, stride):
        grid            = self.grids[k]
        hsize, wsize    = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv          = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid            = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k]   = grid
        grid                = grid.view(1, -1, 2)  # [1, 6400, 2]: [[[ 0.,  0.], [ 1.,  0.], [ 2.,  0.], ..., [77., 79.], [78., 79.], [79., 79.]]]

        # 此处要修改为bbox TODO
        output              = output.flatten(start_dim=2).permute(0, 2, 1) # [24, 6400, 25] output[..., :2]: {-4.0080, 4.7496} output[..., 2:4]: {-2.2407, 7.0197}
        output[..., :2]     = (output[..., :2] + grid.type_as(output)) * stride  # pred-xy scale到当前featuremap尺寸 [24, 6400, 25] output[..., :2]: {-32.0639, 637.6134}
        output[..., 2:4]    = torch.exp(output[..., 2:4]) * stride  # pred-wh scale到当前featuremap尺寸 # [24, 6400, 25] output[..., 2:4]: {0.8511, 8947.4336}
        if self.ex_kpts>0:
            output[..., 5+self.num_classes::3]     = (output[..., 5+self.num_classes::3] + grid[:, :, 0:1].type_as(output)) * stride  # pred-kptx scale到当前featuremap尺寸
            output[..., 5+1+self.num_classes::3]     = (output[..., 5+1+self.num_classes::3] + grid[:, :, 1:2].type_as(output)) * stride  # pred-kpty scale到当前featuremap尺寸
            
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        bbox_preds  = outputs[:, :, :4]  #   [batch, n_anchors_all, 4]: [24, 8400, 4] {-32.0639, 8947.4336}
        obj_preds   = outputs[:, :, 4:5]#   [batch, n_anchors_all, 1]: [24, 8400, 1] {-29.6609, 6.9036}
        cls_preds   = outputs[:, :, 5:self.num_classes+5]  #   [batch, n_anchors_all, n_cls]: [24, 8400, 20] {-25.3338, 2.6940}
        if self.ex_kpts>0:
            # [x, y, w, h, obj, clses, kpts]
            # [0, 1, 2, 3, 4,   5,     6,7,8, ...]
            kpts_cls_preds   = outputs[:, :, self.num_classes+5+2::3]  # TODO 检查这里是不是？？？
            kpts_reg_preds   = torch.cat([outputs[:, :, self.num_classes+5::3], outputs[:, :, self.num_classes+5+1::3]], dim=2)

        total_num_anchors   = outputs.shape[1]  # 8400  这里实际上是当前层gridcell个数，也就是gridcell-based
        #-----------------------------------------------#
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        #-----------------------------------------------#
        x_shifts            = torch.cat(x_shifts, 1).type_as(outputs) # [1, 8400] {0, 79}
        y_shifts            = torch.cat(y_shifts, 1).type_as(outputs) # [1, 8400] {0, 79}
        expanded_strides    = torch.cat(expanded_strides, 1).type_as(outputs) # [1, 8400] {8, 32}
        
        cls_targets = []
        reg_targets = []
        obj_targets = []
        kpts_cls_targets = []
        kpts_cls_targets_ = []
        kpts_reg_targets = []
        fg_masks    = []

        num_fg  = 0.0
        num_gts = 0
        for batch_idx in range(outputs.shape[0]):
            # 计算batch中每张图片
            num_gt          = len(labels[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target  = outputs.new_zeros((0, self.num_classes))
                reg_target  = outputs.new_zeros((0, 4))
                obj_target  = outputs.new_zeros((total_num_anchors, 1))
                fg_mask     = outputs.new_zeros(total_num_anchors).bool()
                if self.ex_kpts>0:
                    kpts_cls_target  = outputs.new_zeros((0, self.ex_kpts))
                    kpts_reg_target  = outputs.new_zeros((0, self.ex_kpts*2))
            else:
                #-----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, num_classes]
                #   gt_classes              [num_gt]
                #   bboxes_preds_per_image  [n_anchors_all, 4]
                #   cls_preds_per_image     [n_anchors_all, num_classes]
                #   obj_preds_per_image     [n_anchors_all, 1]
                #-----------------------------------------------#
                gt_bboxes_per_image     = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes              = labels[batch_idx][..., 4].type_as(outputs)
                bboxes_preds_per_image  = bbox_preds[batch_idx]
                cls_preds_per_image     = cls_preds[batch_idx]
                obj_preds_per_image     = obj_preds[batch_idx]
                # if self.ex_kpts>0:
                #     gt_kpts_per_image     = labels[batch_idx][..., 5:].type_as(outputs)
                #     kpts_preds_per_image     = kpts_preds[batch_idx]

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments( 
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts, 
                )
                torch.cuda.empty_cache()
                
                num_fg      += num_fg_img
                cls_target  = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target  = fg_mask.unsqueeze(-1)
                reg_target  = gt_bboxes_per_image[matched_gt_inds]
                
                if self.ex_kpts>0:
                    # 注意：这里必须用matched_gt_inds进行匹配
                    kpts_cls_target     = labels[batch_idx][..., 5+2::3][matched_gt_inds].type_as(outputs) # [1, 17]: {0., 2.}
                    # kpts_reg_target     = torch.cat([labels[batch_idx][..., 5::3][matched_gt_inds], labels[batch_idx][..., 5+1::3][matched_gt_inds]], dim=1).type_as(outputs) # [1, 34]: {0., 501.}
                    # match_grid_x = x_shifts[fg_mask[None, :]]
                    # match_grid_y = y_shifts[fg_mask[None, :]]
                    # TODO 这里的640为input shape，因此可以在dataset中做  kpts_reg_target: [38, 34]
                    # kpts_reg_x = labels[batch_idx][..., 5::3][matched_gt_inds]/expanded_strides[fg_mask[None, :]][:, None] - match_grid_x[:, None]
                    # kpts_reg_y = labels[batch_idx][..., 6::3][matched_gt_inds]/expanded_strides[fg_mask[None, :]][:, None] - match_grid_y[:, None]
                    kpts_reg_x = labels[batch_idx][..., 5::3][matched_gt_inds]
                    kpts_reg_y = labels[batch_idx][..., 6::3][matched_gt_inds]
                    kpts_reg_target     = torch.cat([kpts_reg_x, kpts_reg_y], dim=1).type_as(outputs) # [1, 34]: {0., 501.}
                
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)
            if self.ex_kpts>0:
                kpts_cls_targets.append(kpts_cls_target)
                kpts_reg_targets.append(kpts_reg_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks    = torch.cat(fg_masks, 0)
        if self.ex_kpts>0:
            kpts_cls_targets = torch.cat(kpts_cls_targets, 0)
            kpts_reg_targets = torch.cat(kpts_reg_targets, 0)
            
        num_fg      = max(num_fg, 1)
        # compute loss
        loss_iou    = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj    = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls    = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        loss_kpts_score, loss_kpts_reg = torch.zeros(size=[1]).type_as(loss_iou), torch.zeros(size=[1]).type_as(loss_iou)
        
        if self.ex_kpts > 0:
            #
            #
            # 现在kpts_cls_preds.shape=[B, 8400, 17], kpts_cls_targets.shape=[45, 17]，这个需要使用bcewithlog_loss，
            # 如：obj_preds.view(-1, 1).shape=torch.Size([201600, 1])/obj_targets.shape=torch.Size([201600, 1])
            # 需要保持尺寸一致，找到对应gridcell
            #
            #
            with_neg = False
            kpts_cls_mask = kpts_cls_targets > 0
            kpts_cls_targets[kpts_cls_mask] = 1  # [42, 17]
            kpts_cls_preds = kpts_cls_preds.view(-1, self.ex_kpts)  # [16800, 17]
            kpts_cls_preds = kpts_cls_preds[fg_masks, :]  # 正样本  # [42, 17]
            kpts_cls_preds = kpts_cls_preds.view(-1, 1)  # [714, 1] kpts_cls_preds.sum()=22286.7793  kpts_cls_preds.max()=374.9816
            # kpts_cls_preds_ = kpts_cls_preds_.view(-1, 1) # [284886, 1]  kpts_cls_preds_.sum()=67337648.  kpts_cls_preds_.max()=659.5246  kpts_cls_targets.max()=1.
            loss_kpts_cls    = self.bcewithlog_loss(kpts_cls_preds, kpts_cls_targets.view(-1, 1))
            # if with_neg:
            #     loss_kpts_cls_    = self.bcewithlog_loss(kpts_cls_preds_, torch.zeros_like(kpts_cls_preds_))
            # loss_kpts_cls = loss_kpts_cls.view(-1, self.ex_kpts)[kpts_cls_mask].sum()  # 是否平均到每个点???
            loss_kpts_cls = (loss_kpts_cls.view(-1, self.ex_kpts)*kpts_cls_mask.int()).sum()  # 是否平均到每个点???
            # loss_kpts_cls_pos = loss_kpts_cls.view(-1, self.ex_kpts).sum()  # 是否平均到每个点???  loss_kpts_cls.mean()=21.9739
            # if with_neg:
            #     loss_kpts_cls_neg_2 = loss_kpts_cls_.view(-1, self.ex_kpts).sum()  # 是否平均到每个点???  loss_kpts_cls_.mean()=236.4359
            
            kpts_reg_preds = kpts_reg_preds.view(-1, self.ex_kpts*2)  # kpts_reg_preds: [2, 8400, 34]
            kpts_reg_preds_pos = kpts_reg_preds[fg_masks, :]  # kpts_reg_preds: [45, 34]
            #  d: [45, 17]  s: [45, 1]  kpt_loss_factor: 1.9669  k: [17]
            # d = (kpts_reg_targets[:, 0:self.ex_kpts] - kpts_reg_preds_pos[:, 0:self.ex_kpts])**2 + (kpts_reg_targets[:, self.ex_kpts:] - kpts_reg_preds_pos[:, self.ex_kpts:])**2
            # s = torch.prod(reg_targets[:,-2:], dim=1, keepdim=True)#**2
            # kpt_loss_factor = (torch.sum(kpts_cls_mask != 0) + torch.sum(kpts_cls_mask == 0))/torch.sum(kpts_cls_mask != 0)
            # # k = (torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]).type_as(kpts_reg_targets) / 10.0)**2
            # # k = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]).type_as(kpts_reg_targets)
            # if False:
            #     loss_kpts_reg_pos = kpt_loss_factor*(1 - torch.exp(-d/(2*s*k+1e-9)))
            # if True:
            #     loss_kpts_reg_pos = kpt_loss_factor * d / s # k * d / s
            # d = ((kpts_reg_targets[:, 0:self.ex_kpts] - kpts_reg_preds_pos[:, 0:self.ex_kpts])/reg_targets[:,2:3])**2 + ((kpts_reg_targets[:, self.ex_kpts:] - kpts_reg_preds_pos[:, self.ex_kpts:])/reg_targets[:,3:4])**2
            d = torch.abs(kpts_reg_targets[:, 0:self.ex_kpts] - kpts_reg_preds_pos[:, 0:self.ex_kpts])/reg_targets[:,2:3] + torch.abs(kpts_reg_targets[:, self.ex_kpts:] - kpts_reg_preds_pos[:, self.ex_kpts:])/reg_targets[:,3:4]
            # kpt_loss_factor = (torch.sum(kpts_cls_mask != 0) + torch.sum(kpts_cls_mask == 0))/torch.sum(kpts_cls_mask != 0)
            # # k = (torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]).type_as(kpts_reg_targets) / 10.0)**2
            # k = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]).type_as(kpts_reg_targets)
            loss_kpts_reg_pos = d  # kpt_loss_factor * k * d
            loss_kpts_reg_pos = (loss_kpts_reg_pos*kpts_cls_mask.int()).mean(dim=1).sum()
            
            loss_kpts_score = loss_kpts_cls
            loss_kpts_reg = loss_kpts_reg_pos

        reg_weight  = 5.0
        loss_iou = reg_weight * loss_iou
        loss_obj = 0.1*loss_obj
        loss_kpts_score = 10*loss_kpts_score
        loss_kpts_reg = 10*loss_kpts_reg
        loss = loss_iou + loss_obj + loss_cls + loss_kpts_score + loss_kpts_reg

        return loss / num_fg, {
            'iou': loss_iou.item()/num_fg, 
            'obj': loss_obj.item()/num_fg, 
            'cls': loss_cls.item()/num_fg, 
            'kpts-cls': loss_kpts_score.item()/num_fg, 
            'kpts-reg': loss_kpts_reg.item()/num_fg
        }

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        '''
        total_num_anchors, 
        gt: num_gt, gt_bboxes_per_image, gt_classes  [num_gt, 4/1/1]
        pred: bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image
        mask: expanded_strides, x_shifts, y_shifts  xy_shift实际上就是grid拉平了
        '''
        #-------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, len(fg_mask)]
        #-------------------------------------------------------#
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)

        #-------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes]
        #   obj_preds_              [fg_mask, 1]
        #-------------------------------------------------------#
        bboxes_preds_per_image  = bboxes_preds_per_image[fg_mask]
        cls_preds_              = cls_preds_per_image[fg_mask]
        obj_preds_              = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor     = bboxes_preds_per_image.shape[0]

        #-------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask]
        #-------------------------------------------------------#
        pair_wise_ious      = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        #-------------------------------------------------------#
        #   cls_preds_          [num_gt, fg_mask, num_classes]
        #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
        #-------------------------------------------------------#
        cls_preds_          = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        gt_cls_per_image    = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        pair_wise_cls_loss  = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_

        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg
    
    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt, center_radius = 2.5):
        #-------------------------------------------------------#
        #   expanded_strides_per_image  [n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #-------------------------------------------------------#
        expanded_strides_per_image  = expanded_strides[0]
        x_centers_per_image         = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image         = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        #-------------------------------------------------------#
        #   gt_bboxes_per_image_x       [num_gt, n_anchors_all]
        #-------------------------------------------------------#
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)

        #-------------------------------------------------------#
        #   bbox_deltas     [num_gt, n_anchors_all, 4]
        #-------------------------------------------------------#
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        #-------------------------------------------------------#
        #   is_in_boxes     [num_gt, n_anchors_all]
        #   is_in_boxes_all [n_anchors_all]
        #-------------------------------------------------------#
        is_in_boxes     = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)

        #-------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]
        #-------------------------------------------------------#
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas       = torch.stack([c_l, c_t, c_r, c_b], 2)

        #-------------------------------------------------------#
        #   is_in_centers       [num_gt, n_anchors_all]
        #   is_in_centers_all   [n_anchors_all]
        #-------------------------------------------------------#
        is_in_centers       = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all   = is_in_centers.sum(dim=0) > 0

        #-------------------------------------------------------#
        #   is_in_boxes_anchor      [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        #-------------------------------------------------------#
        is_in_boxes_anchor      = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center  = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        #-------------------------------------------------------#
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]        
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        #-------------------------------------------------------#
        matching_matrix         = torch.zeros_like(cost)

        #------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        #------------------------------------------------------------#
        n_candidate_k           = min(10, pair_wise_ious.size(1))
        topk_ious, _            = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks              = torch.clamp(topk_ious.sum(1).int(), min=1)
        
        for gt_idx in range(num_gt):
            #------------------------------------------------------------#
            #   给每个真实框选取最小的动态k个点
            #------------------------------------------------------------#
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        #------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        #------------------------------------------------------------#
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            #------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。
            #------------------------------------------------------------#
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        #------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        #------------------------------------------------------------#
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg          = fg_mask_inboxes.sum().item()

        #------------------------------------------------------------#
        #   对fg_mask进行更新
        #------------------------------------------------------------#
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        #------------------------------------------------------------#
        #   获得特征点对应的物品种类
        #------------------------------------------------------------#
        matched_gt_inds     = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes  = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
