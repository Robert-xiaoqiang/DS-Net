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

from ..helper.TrainHelper import AverageMeter, LoggerPather, DeviceWrapper, FullModel
from ..helper.TestHelper import Evaluator
from ..inference.Deducer import Deducer

class SupervisedTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloaders, config):
        self.model = model
        self.config = config

        cudnn.benchmark = self.config.CUDNN.BENCHMARK
        cudnn.deterministic = self.config.CUDNN.DETERMINISTIC
        cudnn.enabled = self.config.CUDNN.ENABLED

        self.train_dataloader = train_dataloader
        self.num_train_batch_per_epoch = len(self.train_dataloader)
        self.num_epochs = self.config.TRAIN.NUM_EPOCHS
        self.num_iterations = self.num_epochs * self.num_train_batch_per_epoch

        self.val_dataloader = val_dataloader
        self.test_dataloaders = test_dataloaders

        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer() # build self.num_iterations firstly
        self.lr_scheduler = self.build_lr_scheduler()

        self.wrapped_device = DeviceWrapper()(config.DEVICE)
        self.main_device = torch.device(self.wrapped_device if self.wrapped_device == 'cpu' else 'cuda:' + str(self.wrapped_device[0]))

        self.vanilla_model = self.model      
        self.wrap_model()

        loggerpather = LoggerPather(self.config)
        self.logger = loggerpather.get_logger()
        self.snapshot_path = loggerpather.get_snapshot_path()
        self.tb_path = loggerpather.get_tb_path()
        self.prediction_path = loggerpather.get_prediction_path()
        self.writer = SummaryWriter(self.tb_path)
        self.loss_avg_meter = AverageMeter()
        
        self.loaded_epoch = None
        self.to_pil = transforms.ToPILImage()
        self.best_val_results = {
            'MAE': 2147483647.0,
            # 'MAXE': 0.0,
            'S': 0.0,
            # 'MAXF': 0.0,
        }

    def wrap_model(self):
        self.model = FullModel(self.model, self.criterion)
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.to(self.main_device)
        if type(self.wrapped_device) == list:
            self.model = nn.DataParallel(self.model, device_ids = self.wrapped_device)        

    def build_criterion(self):
        criterion = nn.BCELoss(reduction=self.config.TRAIN.REDUCTION)
        return criterion

    def build_optimizer(self):
        if self.config.TRAIN.OPTIM == "sgd_trick":
            # https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_tricks.py
            params = [
                {
                    "params": [
                        p for name, p in self.model.named_parameters()
                        if ("bias" in name or "bn" in name)
                    ],
                    "weight_decay":
                        0,
                },
                {
                    "params": [
                        p for name, p in self.model.named_parameters()
                        if ("bias" not in name and "bn" not in name)
                    ]
                },
            ]
            optimizer = SGD(
                params,
                lr=self.config.TRAIN.LR,
                momentum=self.config.TRAIN.MOMENTUM,
                weight_decay=self.config.TRAIN.WD,
                nesterov=self.config.TRAIN.NESTEROV
            )
        elif self.config.TRAIN.OPTIM == "sgd_r3":
            params = [
                # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
                # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
                # 到减少模型过拟合的效果。
                {
                    "params":
                        [param for name, param in self.model.named_parameters() if
                         name[-4:] == "bias"],
                    "lr":
                        2 * self.config.TRAIN.LR,
                },
                {
                    "params":
                        [param for name, param in self.model.named_parameters() if
                         name[-4:] != "bias"],
                    "lr":
                        self.config.TRAIN.LR,
                    "weight_decay":
                        self.config.TRAIN.WD,
                },
            ]
            optimizer = SGD(params, momentum=self.config.TRAIN.MOMENTUM)
        elif self.config.TRAIN.OPTIM == "sgd_all":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.config.TRAIN.LR,
                weight_decay=self.config.TRAIN.WD,
                momentum=self.config.TRAIN.MOMENTUM
            )
        elif self.config.TRAIN.OPTIM == "adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=self.config.TRAIN.LR,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.TRAIN.WD
            )
        else:
            raise NotImplementedError
        return optimizer
       
    def build_lr_scheduler(self):
        lamb = lambda curr: pow((1 - float(curr) / self.num_iterations), self.config.TRAIN.LD)
        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lamb)
        return scheduler

    def build_train_model(self):
        b = False
        if self.config.TRAIN.RESUME:
            b = self.load_checkpoint(snapshot_key = 'latest')
            self.logger.info('loaded successfully, continue training from epoch {}'.format(self.loaded_epoch) \
            if b else 'loaded failed, train from ImageNet scratch')
        if not b:
            self.logger.info('loaded nothing new')

    def multigpu_heuristic(self, state_dict):
        new_state_dict = OrderedDict()
        curr_state_dict_keys = set(self.model.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for key, value in state_dict.items():
            new_key = key
            if key not in curr_state_dict_keys:
                if key.startswith('module.'):
                    new_key = key[7:] # load distributed model to single gpu
                else:
                    new_key = 'module.' + key # load model to multi gpus
            if new_key in curr_state_dict_keys:
                new_state_dict[new_key] = value
            else:
                self.logger.info('there are unknown keys in loaded checkpoint')
        return new_state_dict

    def load_checkpoint(self, snapshot_key = 'latest'):
        '''
            load checkpoint and
            make self.loaded_epoch
        '''
        model_file_name = os.path.join(self.snapshot_path, 'model_{}.ckpt'.format(snapshot_key))
        if not os.path.isfile(model_file_name):
            self.logger.info('Cannot find pretrained model checkpoint: ' + model_file_name)
            return False
        else:
            self.logger.info('Find pretrained model checkpoint successfully: ' + model_file_name)            
            map_location = (lambda storage, loc: storage) if self.main_device == 'cpu' else self.main_device
            params = torch.load(model_file_name, map_location = map_location)
            
            model_state_dict = params['model_state_dict']
            model_state_dict = self.multigpu_heuristic(model_state_dict)
            self.model.load_state_dict(model_state_dict)
        
            self.optimizer.load_state_dict(params['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(params['lr_scheduler_state_dict'])
            self.loaded_epoch = params['epoch']
            return True

    def save_checkpoint(self, epoch, snapshot_key = 'latest'):
        self.summary_model(epoch, snapshot_key)
        # if snapshot_key != 'latest':
        #     self.summary_model(epoch, 'latest')

    # epoch to resume after suspending or storing
    def summary_model(self, epoch, snapshot_key = 'latest'):
        model_file_name = os.path.join(self.snapshot_path, 'model_{}.ckpt'.format(snapshot_key))
        torch.save({ 'model_state_dict': self.model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict(),
                     'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                     'epoch': epoch
                    }, model_file_name)
        self.logger.info('save model in {}'.format(model_file_name))

    def build_data(self, batch_data):
        batch_data = [ d.to(self.main_device, non_blocking=True) if torch.is_tensor(d) else d for d in batch_data ]
        return tuple(batch_data)

    def train_epoch(self, epoch):
        # set evaluation mode in self.on_epoch_end()
        self.model.train()
        for batch_index, batch_data in enumerate(self.train_dataloader):
            batch_rgb, batch_label\
            = self.build_data(batch_data)

            losses, output = self.model(batch_rgb, batch_label)
            # here loss is gathered from each rank, mean/sum it to scalar
            if self.config.TRAIN.REDUCTION == 'mean':
                loss = losses.mean()
            else:
                loss = losses.sum()
            self.on_batch_end(output, batch_label, loss, epoch, batch_index)

    def on_batch_end(self, output, batch_label, loss,
                        epoch, batch_index):
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        iteration = epoch * self.num_train_batch_per_epoch + batch_index + 1

        self.loss_avg_meter.update(loss.item())

        if not iteration % self.config.TRAIN.LOSS_FREQ:
            self.summary_loss(loss, epoch, iteration)
        
        if not iteration % self.config.TRAIN.TB_FREQ:
            self.summary_tb(output, batch_label, loss, epoch, iteration)

    def summary_tb(self, output, batch_label, loss, epoch, iteration):
        train_batch_size = output.shape[0]
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], iteration)
        self.writer.add_scalar('train/loss_cur', loss, iteration)
        self.writer.add_scalar('train/loss_avg', self.loss_avg_meter.average(), iteration)
        
        tr_tb_mask = make_grid(batch_label[:8], nrow=8, padding=5)
        self.writer.add_image('train/masks', tr_tb_mask, iteration)
        
        tr_tb_out_1 = make_grid(output[:8], nrow=8, padding=5)
        self.writer.add_image('train/preds', tr_tb_out_1, iteration)

    def summary_loss(self, loss, epoch, iteration):
        self.logger.info('[epoch {}/{} - iteration {}/{}]: loss(cur): {:.4f}, loss(avg): {:.4f}, lr: {:.8f}'\
                .format(epoch, self.num_epochs, iteration, self.num_iterations, loss.item(), self.loss_avg_meter.average(), self.optimizer.param_groups[0]['lr']))

    def on_epoch_end(self, epoch):
        self.lr_scheduler.step(epoch + 1)
        self.save_checkpoint(epoch + 1)
        val_loss, results = self.validate()
        
        is_update = results['MAE'] < self.best_val_results['MAE'] and \
                    results['S'] > self.best_val_results['S']
        
        self.writer.add_scalar('val/loss_cur', val_loss, epoch)
        self.writer.add_scalar('val/S', results['S'], epoch)
        # self.writer.add_scalar('val/MAXF', results['MAXF'], epoch)
        # self.writer.add_scalar('val/MAXE', results['MAXE'], epoch)
        self.writer.add_scalar('val/MAE', results['MAE'], epoch)

        if is_update:
            self.best_val_results.update(results)
            self.save_checkpoint(epoch + 1, 'best')
            self.logger.info('Update best epoch')
            self.logger.info('Epoch {} with best validating results: {}'.format(epoch, self.best_val_results))
        else:
            self.logger.info('Epoch with validating loss {:.4f}, without updating best epoch'.format(val_loss))

    def on_train_end(self):
        self.logger.info('Finish training with epoch {}, close all'.format(self.num_epochs))
        self.writer.close()
        self.test()

    def train(self):
        self.build_train_model()
        
        start_epoch = self.loaded_epoch if self.loaded_epoch is not None else 0
        end_epoch = self.num_epochs

        for epoch in range(start_epoch, end_epoch):
            self.train_epoch(epoch)
            self.on_epoch_end(epoch)
        self.on_train_end()

    def test(self):
        deducer = Deducer(self.vanilla_model, self.test_dataloaders, self.config)
        deducer.deduce()

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
                losses, output = self.model(batch_rgb, batch_label)
            
            val_loss.update(losses.mean().item())
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
