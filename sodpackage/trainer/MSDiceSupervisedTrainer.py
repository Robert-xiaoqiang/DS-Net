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

from ..helper.TrainHelper import AverageMeter, LoggerPather, DeviceWrapper, MSDiceFullModel, DiceLoss
from ..helper.TestHelper import Evaluator
from ..inference.Deducer import Deducer

from .SupervisedTrainer import SupervisedTrainer

class MSDiceSupervisedTrainer(SupervisedTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloaders, config):
        super().__init__(model, train_dataloader, val_dataloader, test_dataloaders, config)

    def wrap_model(self):
        self.model = MSDiceFullModel(self.model, self.criterion)
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.to(self.main_device)
        if type(self.wrapped_device) == list:
            self.model = nn.DataParallel(self.model, device_ids = self.wrapped_device)        

    def build_criterion(self):
        criterion = [ nn.BCELoss(reduction = self.config.TRAIN.REDUCTION),
                      DiceLoss(reduction = self.config.TRAIN.REDUCTION) ]
        return criterion

    def train_epoch(self, epoch):
        # set evaluation mode in self.on_epoch_end(), here reset training mode
        self.model.train()
        for batch_index, batch_data in enumerate(self.train_dataloader):
            batch_rgb, batch_depth, batch_label\
            = self.build_data(batch_data)

            losses_output = self.model(batch_rgb, batch_depth, batch_label)
            sep = len(losses_output) // 2
            losses, output = losses_output[:sep], losses_output[sep:]

            # here loss is gathered from each rank, mean/sum it to scalar
            if self.config.TRAIN.REDUCTION == 'mean':
                loss = [ l.mean() for l in losses ]
            else:
                loss = [ l.sum() for l in losses ]
            # stage 4 output as final output
            output = output[0]

            self.on_batch_end(output, batch_label, loss, epoch, batch_index)

    def on_batch_end(self, output, batch_label,
                     loss_list,
                     epoch, batch_index):
        
        # reversed stage loss
        loss = loss_list[0] + 0.8 * loss_list[1] + 0.4 * loss_list[2] + 0.2 * loss_list[3]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        iteration = epoch * self.num_train_batch_per_epoch + batch_index + 1

        self.loss_avg_meter.update(loss.item())

        if not iteration % self.config.TRAIN.LOSS_FREQ:
            self.summary_loss(loss, loss_list, epoch, iteration)
        
        if not iteration % self.config.TRAIN.TB_FREQ:
            self.summary_tb(output, batch_label, loss, loss_list, epoch, iteration)

    def summary_tb(self, output, batch_label, loss, loss_list, epoch, iteration):
        train_batch_size = output.shape[0]
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iteration)
        self.writer.add_scalar('train/loss_cur', loss, iteration)
        self.writer.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)
        for i, stage_loss in enumerate(reversed(loss_list), 1):
            self.writer.add_scalar('train/loss_stage_{}'.format(i), stage_loss, iteration)

        row = min(self.config.TRAIN.TB_ROW, train_batch_size)

        tr_tb_mask = make_grid(batch_label[:row], nrow=row, padding=5)
        self.writer.add_image('train/masks', tr_tb_mask, iteration)
        
        tr_tb_out_1 = make_grid(output[:row], nrow=row, padding=5)
        self.writer.add_image('train/preds', tr_tb_out_1, iteration)

    def summary_loss(self, loss, loss_list, epoch, iteration):
        self.logger.info('[epoch {}/{} - iteration {}/{}]: loss(cur): {:.4f} = [{:.4f}+0.8*{:.4f}+0.4*{:.4f}+0.2*{:.4f}], loss(avg): {:.4f}, lr: {:.8f}'\
                .format(epoch, self.num_epochs, iteration, self.num_iterations, \
                loss.item(), loss_list[0].item(), loss_list[1].item(), loss_list[2].item(), loss_list[3].item(), \
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
                losses_output = self.model(batch_rgb, batch_depth, batch_label)
            
            sep = len(losses_output) // 2
            losses, output = losses_output[:sep], losses_output[sep:]

            # here loss is gathered from each rank, mean/sum it to scalar
            if self.config.TRAIN.REDUCTION == 'mean':
                loss = [ l.mean() for l in losses ]
            else:
                loss = [ l.sum() for l in losses ]
            # stage 4 output as final output
            output = output[0]

            loss = loss[0] + 0.8 * loss[1] + 0.4 * loss[2] + 0.2 * loss[3]

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
