import os, sys
import numpy as np
import cv2
import lightning
import torch
import torchvision
from typing import Any, Union


class TrainModule(lightning.LightningModule):
    def __init__(self, configs, *args: Any, **kwargs: Any) -> None:
        '''
        args/kwargs: 不能包含句柄/函数之类的,不然save_hyperparameters报错。
            https://zhuanlan.zhihu.com/p/452780801
        '''
        # super().__init__(*args, **kwargs)
        super().__init__()
        hyper_params = {k: v for k, v in configs.__dict__.items() if not (k.startswith('__') or k.endswith('__'))}
        hyper_params.update(kwargs)
        self.save_hyperparameters(hyper_params)
        self.configs = configs
    
    def create_model(self):
        ...
    
    def create_loss(self, *args: Any, **kwargs: Any):
        ...


class ModelInference():
    def __init__(self, model=None, decoder=None, configs=None, *args, **kwargs):
        self.model = model
        self.decoder = decoder
        self.configs = configs
    
    def inference(self, input_data, model=None, decoder=None, *args, **kwargs):
        '''
        # forward
        # decoder: iou_th, conf_th, nms_th
        # return img, bboxes, kpts
        bboxes: xywhcs, <N, 6>/<N, 4>
        kpts: xycs, <N, 4>  ==> xysxysxys
        kpts_xys2xycs()
        kpts_xycs2xys()
        return {'input_data': input_data, 'bboxes': bboxes, 'kpts': kpts, 'seg': seg}
        '''
        model_ = self.model if model is None else model
        decoder_ = self.decoder if decoder is None else decoder
        if input_data is None or model_ is None or decoder_ is None:
            return {}
        preds = model_(input_data)
        ret_dict = decoder(preds, input_data, *args, **kwargs)
        return ret_dict.update({'input_data': input_data, 'preds': preds})

        
class PredModule(ModelInference):
    def __init__(self, configs) -> None:
        self.configs = configs
    
    def forward(self, data, model=None):
        model_ = self.model if model is None else model
        preds = model_(data)
        return preds
    
    def input_ops(self, data):
        pass
    
    def decode_preds(self, preds):
        pass

    def pred_image(self, image, model=None, params=None):
        ...
    
    def pred_dir(self, img_dir, output_dir, model=None, params=None):
        '''
        图片：最多包含两个层级
        '''
        ...

    def pred_video(self, cam, output_dir, model=None, params=None):
        '''
        取流: mp4, rtsp
        '''
        ...

    def get_fps(self, configs, *args: Any, **kwargs: Any):
        ...


class EvalModule(ModelInference):
    def __init__(self, model, configs) -> None:
        self.model = model
        self.configs = configs

    def pred_coco(self, root_dir, set_name='train2017', output_dir='./', save_image=True):
        ...

    def COCOmAP(self, coco_gt, coco_dt, img_ids, iou_type='bbox'):
        ...

    def VOCmAP(self, coco_gt, coco_dt, img_ids, iou_type='bbox'):
        ...
    
    def onnx(self, model, input, save_file, input_names, output_names):
        ...


class ModelIO(torch.nn.Module):
    def save_model(self, model, save_path, ddp=False):
        torch.save(model, save_path)
    
    def save_state_dict(self, model, save_path, ddp=False):
        model_ = model
        if ddp:
            model_ = model.module
        torch.save(model_.state_dict(), save_path)
    
    def save(self, model, save_path, mode='state_dict', ddp=False):
        if mode == 'state_dict':
            self.save_state_dict(model, save_path, ddp)
        else:
            self.save_model(model, save_path, ddp)
    
    def load_params(self, model_path, model, map_location=None, strict=False):
        '''
        ****** model_path: 为checkpoint格式 ******
        '''
        return model.load_state_dict(torch.load(model_path, map_location=map_location), strict=strict)
    
    def load_checkpoint(self, model_path, model=None, map_location=None, strict=False):
        '''
        ****** model_path: 为checkpoint格式 ******
        如果model不为None，返回model，此时只有state_dict;
        如果model为None，返回checkpoint，此时包含所有信息。
        '''
        checkpoint = torch.load(model_path, map_location=map_location)
        if model is not None:
            model_ = checkpoint['state_dict']
            # if list(model_.keys())[0].startswith('model.'):
            #     model_ = {k[6:]:v for k, v in model_.items()}
            model.load_state_dict(model_, strict=strict)
            return model
        return checkpoint
    
    def load(self, model_path, mode='state_dict', model=None, device=None, strict=False):
        '''
            device: int, int-list, None
        '''
        if isinstance(device, int):
            map_location=lambda storage, loc: storage.cuda(device)
        elif isinstance(device, list):
            map_location={f'cuda:{device[1]}':f'cuda:{device[0]}'}
        else:
            ## 默认cpu
            # map_location=torch.device('cpu')
            map_location=lambda storage, loc: storage

        if mode == 'state_dict':
            return self.load_params(model_path, model, map_location, strict=strict)
        else:
            return self.load_checkpoint(model_path, model, map_location, strict=strict)


class Model(ModelIO):
    
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def init_weights(net, init_type='normal', init_gain = 0.02):
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
        print('initialize network weights with %s type' % init_type)
        net.apply(init_func)
        print('initialize network weights with %s type' % init_type)

    # def load_checkpoints(self, module, mode='state_dict', device: Union[str, int, list]='cpu', strict=True):
    #     if isinstance(device, int):
    #         map_location=lambda storage, loc: storage.cuda(device)
    #     elif isinstance(device, list):
    #         map_location={f'cuda:{device[1]}':f'cuda:{device[0]}'}
    #     else:
    #         ## 默认cpu
    #         # map_location=torch.device('cpu')
    #         map_location=lambda storage, loc: storage
    #     if mode == 'state_dict':
    #         module_data = self.load_state_dict(torch.load(module, map_location=map_location))
    #         model = module_data
    #     else:
    #         module_data = torch.load(module, map_location=map_location)
    #         model = module_data['state_dict']
    #         if list(model.keys())[0].startswith('model.'):
    #             model = {k[6:]:v for k, v in model.items()}
    #         self.load_state_dict(model, strict=strict)
    #     return self
    
    # def load_pretrained_ckpt(self, model):
    #     module = torch.load(model)
    #     if 'state_dict' in module.keys():
    #         pretrained_dict = module['state_dict']
    #     else:
    #         pretrained_dict = module

    #     model_dict = self.state_dict()
    #     new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    #     model_dict.update(new_dict)
    #     self.load_state_dict(model_dict)
        
    # def save_checkpoints(self, name, save_dir='./ckpts', mode=['state_dict']):
    #     torch.save(self.state_dict(), os.path.join(save_dir, name))
    
    def freeze(self, freeze_layers: Union[str, list, dict]):
        pass
    
    def unfreeze(self, freeze_layers: Union[str, list, dict]=None):
        pass
        





# 
# data: dataset, augument
# 数据建模: Encoder, loss  ==> map precision recall
# 特征工程：net  ==>  feature-map
# optim: 
# 








