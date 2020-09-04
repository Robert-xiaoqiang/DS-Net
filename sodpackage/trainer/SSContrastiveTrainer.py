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

from ..helper.TrainHelper import AverageMeter, LoggerPather, DeviceWrapper, ContrastiveFullModel
from ..helper.TestHelper import Evaluator
from .ContrastiveTrainer import ContrastiveTrainer

class SSContrastiveTrainer(ContrastiveTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloaders, config):
        super().__init__(model, train_dataloader, val_dataloader, test_dataloaders, config)

    def on_batch_end(self, output, batch_label, 
                     supervised_loss, contrastive_loss,
                     epoch, batch_index):
        
        loss = contrastive_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        iteration = epoch * self.num_train_batch_per_epoch + batch_index + 1

        self.loss_avg_meter.update(loss.item())

        if not iteration % self.config.TRAIN.LOSS_FREQ:
            self.summary_loss(loss, contrastive_loss, epoch, iteration)
        
        if not iteration % self.config.TRAIN.TB_FREQ:
            self.summary_tb(output, batch_label, loss, contrastive_loss, epoch, iteration)

    def summary_tb(self, output, batch_label, loss, contrastive_loss, epoch, iteration):
        train_batch_size = output.shape[0]
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iteration)
        self.writer.add_scalar('train/loss_cur', loss, iteration)
        self.writer.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)
        self.writer.add_scalar('train/loss_contrastive', contrastive_loss, iteration)

        tr_tb_mask = make_grid(batch_label[:8], nrow=8, padding=5)
        self.writer.add_image('train/masks', tr_tb_mask, iteration)
        
        tr_tb_out_1 = make_grid(output[:8], nrow=8, padding=5)
        self.writer.add_image('train/preds', tr_tb_out_1, iteration)

    def summary_loss(self, loss, contrastive_loss, epoch, iteration):
        self.logger.info('[epoch {}/{} - iteration {}/{}]: loss(cur): {:.4f} = [{:.4f}+{:.4f}], loss(avg): {:.4f}, lr: {:.8f}'\
                .format(epoch, self.num_epochs, iteration, self.num_iterations, \
                loss.item(), 0.0, contrastive_loss.item(), self.loss_avg_meter.average(), \
                self.optimizer.param_groups[0]['lr']))

    def validate(self):
        self.model.eval()
        val_loss = AverageMeter()

        preds = [ ]
        masks = [ ]
        tqdm_iter = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), leave=False)
        for batch_id, batch_data in tqdm_iter:
            tqdm_iter.set_description(f'Infering: te=>{batch_id + 1}')
            with torch.no_grad():
                batch_rgb, batch_label, batch_mask_path, batch_key, \
                = self.build_data(batch_data)
                losses, contrastives, output = self.model(batch_rgb, batch_label)
            
            val_loss.update(contrastives.mean().item())
            output_cpu = output.cpu().detach()
            for pred, mask_path in zip(output_cpu, batch_mask_path):
                mask = copy.deepcopy(Image.open(mask_path).convert('L'))
                pred = self.to_pil(pred).resize(mask.size)
                preds.append(pred)
                masks.append(mask)
        self.logger.info('Start evaluation on validating dataset')
        results = Evaluator.fast_evaluate(preds, masks)
        self.logger.info('Finish evaluation on validating dataset')
        return val_loss.average(), results