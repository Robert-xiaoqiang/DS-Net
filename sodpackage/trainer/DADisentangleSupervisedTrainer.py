import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.optim import Adam, SGD, lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from PIL import Image
from tqdm import tqdm

import os
import copy
from collections import OrderedDict

from ..helper.TrainHelper import AverageMeter, LoggerPather, DeviceWrapper, DADisentangleFullModel, BerhuLoss
from ..helper.TestHelper import Evaluator
from ..inference.Deducer import Deducer

from .SupervisedTrainer import SupervisedTrainer

class DADisentangleSupervisedTrainer(SupervisedTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloaders, config):
        super().__init__(model, train_dataloader, val_dataloader, test_dataloaders, config)

    def wrap_model(self):
        self.model = DADisentangleFullModel(self.model, self.criterion)
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.to(self.main_device)
        if type(self.wrapped_device) == list:
            self.model = nn.DataParallel(self.model, device_ids = self.wrapped_device)        

    def build_criterion(self):
        criterion = [ nn.BCELoss(reduction = self.config.TRAIN.REDUCTION),
                      BerhuLoss(reduction = self.config.TRAIN.REDUCTION) ]
        return criterion

    def train_epoch(self, epoch):
        # set evaluation mode in self.on_epoch_end(), here reset training mode
        self.model.train()
        for batch_index, batch_data in enumerate(self.train_dataloader):
            batch_rgb, batch_depth, batch_label\
            = self.build_data(batch_data)

            sod_losses, depth_losses, reconstruct_losses, *output = self.model(batch_rgb, batch_depth, batch_label)
            # here loss is gathered from each rank, mean/sum it to scalar
            if self.config.TRAIN.REDUCTION == 'mean':
                sod_loss = sod_losses.mean()
                depth_loss = depth_losses.mean()
                reconstruct_loss = reconstruct_losses.mean()
            else:
                sod_loss = sod_losses.sum()
                depth_loss = depth_losses.sum()
                reconstruct_loss = reconstruct_losses.sum()
            self.on_batch_end(output, batch_label, sod_loss, depth_loss, reconstruct_loss, epoch, batch_index)

    def on_batch_end(self, output, batch_label,
                     sod_loss, depth_loss, reconstruct_loss,
                     epoch, batch_index):
        
        loss = sod_loss + depth_loss + 1.0 * reconstruct_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        iteration = epoch * self.num_train_batch_per_epoch + batch_index + 1

        self.loss_avg_meter.update(loss.item())

        if not iteration % self.config.TRAIN.LOSS_FREQ:
            self.summary_loss(loss, sod_loss, depth_loss, reconstruct_loss, epoch, iteration)
        
        if not iteration % self.config.TRAIN.TB_FREQ:
            self.summary_tb(output, batch_label, loss, sod_loss, depth_loss, reconstruct_loss, epoch, iteration)

    def summary_tb(self, output, batch_label, loss, sod_loss, depth_loss, reconstruct_loss, epoch, iteration):
        # sod output and its batch size
        train_batch_size = output[0].shape[0]
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iteration)
        self.writer.add_scalar('train/loss_cur', loss, iteration)
        self.writer.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)
        self.writer.add_scalar('train/loss_sod', sod_loss, iteration)
        self.writer.add_scalar('train/loss_depth', depth_loss, iteration)
        self.writer.add_scalar('train/loss_reconstruct', reconstruct_loss, iteration)

        row = min(self.config.TRAIN.TB_ROW, train_batch_size)

        tr_tb_mask = make_grid(batch_label[:row], nrow=row, padding=5)
        self.writer.add_image('train/masks', tr_tb_mask, iteration)
        
        tr_tb_out_1 = make_grid(output[0][:row], nrow=row, padding=5)
        self.writer.add_image('train/sod_preds', tr_tb_out_1, iteration)
        tr_tb_out_2 = make_grid(output[1][:row], nrow=row, padding=5)
        self.writer.add_image('train/depth_preds', tr_tb_out_2, iteration)

    def summary_loss(self, loss, sod_loss, depth_loss, reconstruct_loss, epoch, iteration):
        self.logger.info('[epoch {}/{} - iteration {}/{}]: loss(cur): {:.4f} = [{:.4f}+{:.4f}+1.0*{:.4f}], loss(avg): {:.4f}, lr: {:.8f}'\
                .format(epoch, self.num_epochs, iteration, self.num_iterations, \
                loss.item(), sod_loss.item(), depth_loss.item(), reconstruct_loss.item(), \
                self.loss_avg_meter.average(), self.optimizer.param_groups[0]['lr']))

    def validate(self):
        self.model.eval()
        val_loss = AverageMeter()

        preds = [ ]
        masks = [ ]
        tqdm_iter = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), leave=False)
        for batch_id, batch_data in tqdm_iter:
            tqdm_iter.set_description(f'Infering: te=>{batch_id + 1}')
            with torch.no_grad():
                batch_rgb, batch_depth, batch_label, batch_mask_path, batch_key, \
                = self.build_data(batch_data)
                sod_losses, depth_losses, reconstruct_loss, *output = self.model(batch_rgb, batch_depth, batch_label)
            
            if self.config.TRAIN.REDUCTION == 'mean':
                sod_loss = sod_losses.mean()
                depth_loss = depth_losses.mean()
                reconstruct_loss = reconstruct_losses.mean()
            else:
                sod_loss = sod_losses.sum()
                depth_loss = depth_losses.sum()
                reconstruct_loss = reconstruct_losses.sum()
            loss = sod_loss + depth_loss + 1.81E-5 * reconstruct_loss
            # sod output as final output
            output = output[0]

            val_loss.update(loss.item())
            output_cpu = output.cpu().detach()
            for pred, mask_path in zip(output_cpu, batch_mask_path):
                mask = copy.deepcopy(Image.open(mask_path).convert('L'))
                pred = self.to_pil(pred).resize(mask.size)
                preds.append(pred)
                masks.append(mask)
        self.logger.info('Start evaluation on validating dataset')
        results = Evaluator.evaluate(preds, masks)
        self.logger.info('Finish evaluation on validating dataset')
        return val_loss.average(), results
