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

from ..helper.TrainHelper import AverageMeter, LoggerPather, DeviceWrapper, \
MTFullModel, MTFakeFullModel, BerhuLoss, Ramper
from ..helper.TestHelper import Evaluator
from ..inference.Deducer import Deducer

from .SupervisedTrainer import SupervisedTrainer

class MTSemiSupervisedTrainer(SupervisedTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloaders, config):
        # polymorphism in constructor, let's build some attributes before it !!!!
        self.ema_model = copy.deepcopy(model)
        super().__init__(model, train_dataloader, val_dataloader, test_dataloaders, config)

    def wrap_model(self):
        self.model = MTFullModel(self.model, self.criterion)
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.to(self.main_device)

        self.ema_model = MTFakeFullModel(self.ema_model) # ema_model has no loss and gradient
        self.ema_model.to(self.main_device)
        
        if type(self.wrapped_device) == list:
            self.model = nn.DataParallel(self.model, device_ids = self.wrapped_device)
            self.ema_model = nn.DataParallel(self.ema_model, device_ids = self.wrapped_device)        
        
        # freeze ema_model
        for param in self.ema_model.parameters():
            param.detach_()

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.config.TRAIN.UNLABELED.CONSISTENCY * Ramper.sigmoid_rampup(epoch, self.config.TRAIN.UNLABELED.CONSISTENCY_RAMPUP)
    
    def update_ema_variables(self, global_step):
        # Use the true average until the exponential average is more correct
        alpha = self.config.TRAIN.UNLABELED.EMA_DECAY
        alpha = min(1 - 1 / (global_step + 1), alpha)

        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            # update params in current device (cpu/cuda:)
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
    def build_criterion(self):
        # later `to operation` within fullmodel
        # sod loss for labeled
        # sod loss for unlabeled
        # depth loss for unlabeled
        criterion = [ nn.BCELoss(reduction=self.config.TRAIN.REDUCTION),
                      nn.MSELoss(reduction=self.config.TRAIN.REDUCTION),
                      nn.MSELoss(reduction=self.config.TRAIN.REDUCTION)
                    ]
        return criterion

    def load_checkpoint(self, snapshot_key = 'latest'):
        '''
            load checkpoint and
            make self.loaded_epoch
        '''
        model_file_name = os.path.join(self.snapshot_path, 'model_{}.ckpt'.format(snapshot_key))
        if not os.path.isfile(model_file_name):
            self.logger.info('Cannot find suspended model checkpoint: ' + model_file_name)
            return False
        else:
            self.logger.info('Find suspended model checkpoint successfully: ' + model_file_name)            
            map_location = (lambda storage, loc: storage) if self.main_device == 'cpu' else self.main_device
            params = torch.load(model_file_name, map_location = map_location)
            
            model_state_dict = params['model_state_dict']
            model_state_dict = self.multigpu_heuristic(model_state_dict)
            self.model.load_state_dict(model_state_dict)

            ema_model_state_dict = params['ema_model_state_dict']
            ema_model_state_dict = self.multigpu_heuristic(ema_model_state_dict)
            self.ema_model.load_state_dict(ema_model_state_dict)

            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(params['lr_scheduler_state_dict'])
            self.loaded_epoch = params['epoch']
            return True

    # epoch to resume after suspending or storing
    def summary_model(self, epoch, snapshot_key = 'latest'):
        model_file_name = os.path.join(self.snapshot_path, 'model_{}.ckpt'.format(snapshot_key))
        torch.save({ 'model_state_dict': self.model.state_dict(),
                     'ema_model_state_dict': self.ema_model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict(),
                     'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                     'epoch': epoch
                    }, model_file_name)
        self.logger.info('save model in {}'.format(model_file_name))

    def train_epoch(self, epoch):
        # set evaluation mode in self.on_epoch_end(), here reset training mode
        self.model.train()
        for batch_index, batch_data in enumerate(self.train_dataloader):
            batch_rgb, batch_depth, batch_label\
            = self.build_data(batch_data)
            lb = self.config.TRAIN.BATCH_SIZE - self.config.TRAIN.UNLABELED.BATCH_SIZE
            unlabeled_ema_output = self.ema_model(batch_rgb[lb:], batch_depth[lb:])
            lb_within_card = lb // len(self.wrapped_device)
            supervised_losses, consistency_losses, *output = self.model(batch_rgb, batch_depth, batch_label, unlabeled_ema_output, lb_within_card)
            # here loss is gathered from each rank, mean/sum it to scalar
            if self.config.TRAIN.REDUCTION == 'mean':
                supervised_loss = supervised_losses.mean()
                consistency_loss = consistency_losses.mean()
            else:
                supervised_loss = supervised_losses.sum()
                consistency_loss = consistency_losses.sum()
    
            self.on_batch_end(output, unlabeled_ema_output, batch_label, supervised_loss, consistency_loss, epoch, batch_index)


    def on_batch_end(self, output, unlabeled_ema_output, batch_label,
                     supervised_loss, consistency_loss,
                     epoch, batch_index):
        
        consistency_weight = self.get_current_consistency_weight(epoch)
        loss = supervised_loss + consistency_weight * consistency_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        iteration = epoch * self.num_train_batch_per_epoch + batch_index + 1
        
        self.update_ema_variables(iteration)

        self.loss_avg_meter.update(loss.item())

        if not iteration % self.config.TRAIN.LOSS_FREQ:
            self.summary_loss(loss, supervised_loss, consistency_weight, consistency_loss, epoch, iteration)
        
        if not iteration % self.config.TRAIN.TB_FREQ:
            self.summary_tb(output, unlabeled_ema_output, batch_label, loss, supervised_loss, consistency_weight, consistency_loss, epoch, iteration)

    def summary_tb(self, output, unlabeled_ema_output, batch_label,
                   loss, supervised_loss, consistency_weight, consistency_loss, 
                   epoch, iteration):
        train_batch_size = output[0].shape[0]
        train_unlabeled_batch_size = unlabeled_ema_output[0].shape[0]
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iteration)
        self.writer.add_scalar('train/loss_cur', loss, iteration)
        self.writer.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)
        self.writer.add_scalar('train/loss_supervised', supervised_loss, iteration)
        self.writer.add_scalar('train/consistency_weight', consistency_weight, iteration)
        self.writer.add_scalar('train/loss_consistency', consistency_loss, iteration)
        
        row = min(self.config.TRAIN.TB_ROW, train_batch_size, train_unlabeled_batch_size)

        tr_tb_mask = make_grid(batch_label[:row], nrow=row, padding=5)
        self.writer.add_image('train/masks', tr_tb_mask, iteration)
        
        tr_tb_out_1 = make_grid(output[0][:row], nrow=row, padding=5)
        self.writer.add_image('train/sod_preds', tr_tb_out_1, iteration)

        tr_tb_out_2 = make_grid(output[1][:row], nrow=row, padding=5)
        self.writer.add_image('train/depth_preds', tr_tb_out_2, iteration)

        tr_tb_out_3 = make_grid(unlabeled_ema_output[0][:row], nrow=row, padding=5)
        self.writer.add_image('train/unlabeled/ema_sod_preds', tr_tb_out_3, iteration)

        tr_tb_out_4 = make_grid(unlabeled_ema_output[1][:row], nrow=row, padding=5)
        self.writer.add_image('train/unlabeled/ema_depth_preds', tr_tb_out_4, iteration)

    def summary_loss(self, loss, supervised_loss, consistency_weight, consistency_loss, epoch, iteration):
        self.logger.info('[epoch {}/{} - iteration {}/{}]: loss(cur): {:.4f} = [{:.4f}+{:.4f}*{:.4f}], loss(avg): {:.4f}, lr: {:.8f}'\
                .format(epoch, self.num_epochs, iteration, self.num_iterations,
                        loss.item(), supervised_loss.item(), consistency_weight, consistency_loss.item(), self.loss_avg_meter.average(),
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
                batch_rgb, batch_depth, batch_label, batch_mask_path, batch_key, \
                = self.build_data(batch_data)
                sod_losses, *output = self.model(batch_rgb, batch_depth, batch_label, None, None, is_train = False)
            
            if self.config.TRAIN.REDUCTION == 'mean':
                sod_loss = sod_losses.mean()
            else:
                sod_loss = sod_losses.sum()
            loss = sod_loss
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
