#! /usr/bin/env python
# coding: utf-8
import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torchvision
import torchviz
import lightning as L
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, Optional, EPOCH_OUTPUT
from torch.utils.data.dataloader import DataLoader
# from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pycocotools.coco import COCO

# from dataset.coco_dataset_yolox import COCODetDataset
from dataset.coco_dataset_yolox_v2 import COCODetDataset
# from utils.interface_v2 import TrainModule
from tools.pytorch.interface_v3 import TrainModule
from net.yolo import YoloBody
from loss.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init)
import argparse
from tools.viz.print import show_dict
from tools.viz.pil_draw import PILDraw
from PIL import Image


class YOLOTrainModule(TrainModule):
    def __init__(
        self,
        configs=None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(configs, *args, **kwargs)
        # self.save_hyperparameters({k: v for k, v in configs.__dict__.items() if not (k.startswith('__') or k.endswith('__'))})
        self.model = None
        self.model_ema = None
        self.configs = configs
        self.pred_model = None
        self.example_input_array = torch.FloatTensor(np.random.randint(0, 1, size=(1, 3, self.configs.input_shape[0], self.configs.input_shape[1])))

    def init_configs(self):
        return None

    def create_model(self):
        print(f'loading model ...')
        self.model = YoloBody(self.configs.num_classes, self.configs.phi, ex_kpts=self.configs.ex_kpts)

        if isinstance(self.configs.model_path, str):
            print(f'loading pretrained-model: {self.configs.model_path} ...')
            self.model.load_checkpoint(self.configs.model_path, strict=True)
            # self.model_ema = ModelEmaV2(self.model)
            # self.model_ema = ModelEMA(self.model)
            print(f'loading pretrained-model: {self.configs.model_path} complete.')
        else:
            print(f'init model ...')
            self.model.init_weights(self.model)
            print(f'init model complete.')
        print(f'loading model complete.')

        return self.model

    def create_loss(self):
        self.loss = YOLOLoss(self.configs.num_classes, fp16=self.configs.precision, ex_kpts=self.configs.ex_kpts)

    def forward(self, x):
        return self.model(x)
    
    def debug(self, *args, **kwargs):
        batch_idx = kwargs['batch_idx']
        if batch_idx == 0:
            batch = kwargs['batch']
            images, targets = batch[0], batch[1]
            print(f'')
            imgs_ = []
            for data_idx in range(images.shape[0]):
                img, bbox, label = images[data_idx, ...].cpu().numpy(), targets[data_idx][:, 0:4].cpu().numpy(), targets[data_idx][:, 4].cpu().numpy()
                # print(f'[{batch_idx}/{data_idx}][0]  {bbox[:10, ...]=}')
                # bbox = np.array([box for box in bbox.numpy() if all(box > 0)])
                # print(f'[{batch_idx}/{data_idx}][1]  {bbox[:10, ...]=}')
                img = (np.transpose(img, (1,2,0))*128+127.5).astype(np.uint8)
                bbox[:, 0] -= bbox[:, 2]/2
                bbox[:, 1] -= bbox[:, 3]/2
                bbox[:, 2] += bbox[:, 0]
                bbox[:, 3] += bbox[:, 1]
                bbox = bbox.astype(np.uint32)
                img = PILDraw(font_file=r'./tools/viz/simhei.ttf').rectangle(Image.fromarray(img), bbox, [{'cls': int(l)} for l in label])
                imgs_.append(img)
            imgs_ = np.concatenate(imgs_, axis=1)
            Image.fromarray(imgs_).save(f'./exps/runs/tmp/{self.current_epoch:08}-{batch_idx:08}.jpg')

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        images, targets = batch[0], batch[1]
        batch_size = images.shape[0]

        #----------------------#
        #   计算损失
        #----------------------#
        outputs = self.forward(images)
        loss, loss_items = self.loss(outputs, targets)
        # loss /= batch_size
        
        # ## ema udpate
        # if self.current_epoch > self.configs.EMA_START_EPOCHS:
        #     # self.model_ema.update_parameters(self.model)
        #     # self.scheduler_ema.step()
        #     self.model_ema.update(self.model)
        if self.local_rank == 0:
            self.debug(batch_idx=batch_idx, batch=batch)

        self.log('train-loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log_dict({f'train/{k}': v for k, v in loss_items.items()}, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        images, targets = batch[0], batch[1]
        batch_size = images.shape[0]

        #----------------------#
        #   计算损失
        #----------------------#
        outputs = self.forward(images)
        loss, loss_items = self.loss(outputs, targets)
        
        # loss /= batch_size
        
        self.log('val-loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log_dict({f'val/{k}': v for k, v in loss_items.items()}, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        ## TODO
        ## map进行模型保存依据
        
        ## 统计学习率
        for idx, optim in enumerate(self.optimizer.param_groups):
            self.log(name=f'lr-{idx}', value=optim['lr'], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return super().training_epoch_end(outputs)
    
    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        self.optimizer = {
            'adam'  : optim.Adam(self.model.parameters(), self.configs.init_lr, betas = (self.configs.momentum, 0.999), weight_decay = self.configs.weight_decay),
            'sgd'   : optim.SGD(self.model.parameters(), self.configs.init_lr, momentum = self.configs.momentum, nesterov=True, weight_decay = self.configs.weight_decay)
        }[self.configs.optimizer_type]
        
        self.scheduler = {
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.configs.epochs, verbose=True, eta_min=self.configs.min_lr),
            'step': torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.configs.lr_step_size, gamma=self.configs.gamma, verbose=True),
            'multistep': torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.configs.lr_milestone, gamma=self.configs.gamma, verbose=True),
            'exp': torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.configs.gamma, verbose=True)
        }[self.configs.lr_decay_type]
        
        return [self.optimizer], [{'scheduler': self.scheduler}]


class YoloDataModule(L.LightningDataModule):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
    
    def train_dataloader(self):
        
        ann_file = os.path.join(self.configs['train'].dataset_root, 'annotations', f'instances_{self.configs["train"].dataset_set_name}.json')
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        print(f'load train ann-file: {ann_file} / {len(img_ids)} ...')
        train_dataset = COCODetDataset(
            root_dir=self.configs['train'].dataset_root, 
            set_name=self.configs['train'].dataset_set_name, 
            coco=coco,
            input_shape=self.configs['train'].input_shape, 
            # num_classes=self.configs['train'].num_classes, 
            mosaic=self.configs['train'].mosaic, mosaic_prob=self.configs['train'].mosaic_prob, 
            mixup=self.configs['train'].mixup, mixup_prob=self.configs['train'].mixup_prob, 
            train=self.configs['train'].train, 
            ex_kpts=self.configs['train'].ex_kpts,
            class_list=self.configs['train'].class_list # ['person']
            # configs=self.configs['train']
        )
        train_dataset.check_coco()
        
        # train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        self.train_loader             = DataLoader(
            train_dataset,
            # shuffle = True,
            batch_size = self.configs['train'].batch_size,
            num_workers = self.configs['train'].num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_dataset.yolox_dataset_collate,
            # sampler=train_sampler
        )
        return self.train_loader
    
    def val_dataloader(self):
        ann_file = os.path.join(self.configs['val'].dataset_root, 'annotations', f'instances_{self.configs["val"].dataset_set_name}.json')
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        print(f'load val ann-file: {ann_file} / {len(img_ids)}')
        val_dataset = COCODetDataset(
            root_dir=self.configs['val'].dataset_root, 
            set_name=self.configs['val'].dataset_set_name, 
            coco=coco,
            input_shape=self.configs['val'].input_shape, 
            # num_classes=self.configs['val'].num_classes, 
            mosaic=self.configs['val'].mosaic, mosaic_prob=self.configs['val'].mosaic_prob, 
            mixup=self.configs['val'].mixup, mixup_prob=self.configs['val'].mixup_prob, 
            train=False, 
            ex_kpts=self.configs['val'].ex_kpts,
            class_list=self.configs['train'].class_list
            # configs=self.configs['val']
        )
        val_dataset.check_coco()
        # val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        self.val_loader = DataLoader(
            val_dataset,
            # shuffle=False,
            batch_size=self.configs['val'].batch_size,
            num_workers=self.configs['val'].num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=val_dataset.yolox_dataset_collate,
            # sampler=val_sampler
        )
        return self.val_loader




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--configs', type=str, default=r'./configs/config_instance.py', help='configs path')
    args = parser.parse_args()
    print(f'[main]  load confgis: {args.configs}')
    import imp
    cfg = imp.load_source('b', args.configs)
    show_dict(cfg.TRAINCONFIGS)


    train_module = YOLOTrainModule(configs=cfg.TRAINCONFIGS)
    train_module.create_model()
    train_module.create_loss()

    if cfg.TRAINCONFIGS.LOG_TYPE == 'tb':
        tblogger = TensorBoardLogger(save_dir=cfg.TRAINCONFIGS.LOGS_DIR, log_graph=False)
    elif cfg.TRAINCONFIGS.LOG_TYPE == 'wab':
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        tblogger = WandbLogger(project=f'{cfg.TRAINCONFIGS.EXP_NUM}-{cfg.TRAINCONFIGS.PRE_SYMBOL}')
        tblogger.experiment.config["batch_size"] = cfg.TRAINCONFIGS.batch_size

    checkpoint_cb = ModelCheckpoint(
        dirpath=tblogger.log_dir,
        filename='{epoch:06d}-{train-loss:.4f}-{val-loss:.4f}',
        save_top_k=3,
        monitor=cfg.TRAINCONFIGS.monitor,
        mode=cfg.TRAINCONFIGS.monitor_mode,
        save_last=True,
        save_on_train_epoch_end=False,
        every_n_epochs=1
    )
    data_module = YoloDataModule(configs={'train': cfg.TRAINCONFIGS, 'val': cfg.EVALCONFIGS})
    trainer = L.Trainer(
        logger=tblogger,
        
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        # default_root_dir: Optional[_PATH] = None,
        # gradient_clip_val: Optional[Union[int, float]] = None,
        # gradient_clip_algorithm: Optional[str] = None,
        num_nodes=1,
        # num_processes: Optional[int] = None,  # TODO: Remove in 2.0
        devices=cfg.TRAINCONFIGS.devices,  # 4,
        # gpus: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        # auto_select_gpus: bool = False,
        # tpu_cores: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        # ipus: Optional[int] = None,  # TODO: Remove in 2.0
        enable_progress_bar=True,
        overfit_batches=0.0,
        # track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=1,
        max_epochs=cfg.TRAINCONFIGS.epochs,
        min_epochs=None,
        max_steps=-1,
        min_steps=None,
        max_time=None,
        # limit_train_batches=300,
        # limit_val_batches=300,
        # limit_test_batches=300,
        # limit_predict_batches=300,
        val_check_interval=1.0,
        log_every_n_steps=10,
        accelerator='gpu',
        strategy = "ddp_find_unused_parameters_false",
        sync_batchnorm=cfg.TRAINCONFIGS.sync_bn,
        precision=cfg.TRAINCONFIGS.precision,
        enable_model_summary=True,
        num_sanity_val_steps=10,
        resume_from_checkpoint=None,
        profiler=None,
        # benchmark: Optional[bool] = None,
        # deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        # reload_dataloaders_every_n_epochs: int = 0,
        # auto_lr_find: Union[bool, str] = False,
        # replace_sampler_ddp: bool = True,
        # detect_anomaly: bool = False,
        # auto_scale_batch_size: Union[str, bool] = False,
        # plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        # amp_backend: str = "native",
        # amp_level: Optional[str] = None,
        # move_metrics_to_cpu: bool = False,
        # multiple_trainloader_mode: str = "max_size_cycle",
        # inference_mode: bool = True,
    )
    trainer.fit(train_module, datamodule=data_module)

    # model = CenterNet_Resnet50(80, pretrained = False)
    # import hiddenlayer as h
    # x = torch.randn(1, 3, 512, 512)
    # import PyUtils.pytorch.utils as Utils
    # Utils.write_graph(model, x, filename=f'./model', show_in_nb=False)
