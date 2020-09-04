import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable, Function
import numpy as np

import os
import importlib
import logging
import time
from pathlib import Path

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.double())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss).sum(dim=1)

        # size_average
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        self.eps = 1e-6
    
    def forward(self, pred, target):
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)

class LossCalculator:

    @staticmethod
    def dice_loss(score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss / score.shape[0]

    @staticmethod
    def dice_loss1(score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss / score.shape[0]

    @staticmethod
    def binary_cross_entropy_loss(prediction, target):
        '''
            B * 1 * W * H
            B * 1 * W * H
        '''
        assert prediction.shape == target.shape, 'different shapes of inputs and targets'
        target = target.float() # long -> float
        # bce_model = nn.BCEWithLogitsLoss(reduction='mean').to(torch.cuda.current_device())
        bce_model = nn.BCELoss(reduction='mean').to(torch.cuda.current_device())
        loss = bce_model(prediction, target)
        return loss.squeeze()

    @staticmethod
    def cel_loss(prediction, target):
        '''
            B * 1 * W * H
            B * 1 * W * H
        '''
        assert prediction.shape == target.shape, 'different shapes of inputs and targets'
        target = target.float() # long -> float
        cel_model = CEL().to(torch.cuda.current_device())
        loss = cel_model(prediction, target)
        return loss.squeeze()

    @staticmethod
    def focal_loss(prediction, target):
        '''
            B * 1 * W * H
            B * 1 * W * H            
        '''
        assert prediction.shape == target.shape, 'different shapes of inputs and targets'
        # negate_prediction = torch.ones_like(prediction) - prediction
        target = target.float()
        fl_model = FocalLoss().to(torch.cuda.current_device())
        loss = fl_model(prediction, target)
        return loss.squeeze()

    @staticmethod
    def standard_cross_entropy_loss(score, target):
        '''
            B * 2 * W * H
            B * 1 * W * H
        '''
        assert score.shape[2:] == target.shape[2:], 'different data shapes of inputs and targets'
        ce_model = nn.CrossEntropyLoss(reduction='mean').to(torch.cuda.current_device())
        target = target.squeeze(dim = 1)
        loss = ce_model(score, target) # size_average == True
        return loss.squeeze()
    
    @staticmethod
    def entropy_loss(p,C=2):
        ## p N*C*W*H*D
        y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
        ent = torch.mean(y1)

        return ent

    @staticmethod
    def softmax_dice_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n = input_logits.shape[1]
        dice = 0
        for i in range(0, n):
            dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
        mean_dice = dice / n

        return mean_dice

    @staticmethod
    def entropy_loss_map(p, C=2):
        ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
        return ent

    @staticmethod
    def mse_loss(input_logits, target_logits):
        assert input_logits.size() == target_logits.size()
        target_logits = target_logits.float()
        return F.mse_loss(input_logits, target_logits, reduction = 'sum')

    @staticmethod
    def softmax_mse_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        num_classes = input_logits.size()[1]
        return F.mse_loss(input_softmax, target_softmax, reduction = 'sum') / num_classes

    @staticmethod
    def softmax_kl_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns KL divergence

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        return F.kl_div(input_log_softmax, target_softmax, reduction = 'sum')

    @staticmethod
    def symmetric_mse_loss(input1, input2):
        """Like F.mse_loss but sends gradients to both directions

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to both input1 and input2.
        """
        assert input1.size() == input2.size()
        num_classes = input1.size()[1]
        return torch.sum((input1 - input2)**2) / num_classes

class Ramper:
    @staticmethod
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    @staticmethod
    def linear_rampup(current, rampup_length):
        """Linear rampup"""
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        else:
            return current / rampup_length

    @staticmethod
    def cosine_rampdown(current, rampdown_length):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
        assert 0 <= current <= rampdown_length
        return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

class LoggerPather:
    def __init__(self, cfg):
        # rootpath / experiement_key
        self.root_output_dir = Path(cfg.SUMMARY_DIR) / cfg.NAME
        self.root_output_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.root_output_dir / 'log'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = '{}_{}.log'.format(cfg.NAME, time_str)
        log_file_full_name = self.log_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(log_file_full_name),
                            format=head)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        self.tensorboard_path = self.root_output_dir / 'tensorboard'
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)

        self.snapshot_path = self.root_output_dir / 'snapshot'
        self.snapshot_path.mkdir(parents=True, exist_ok=True)

        self.prediction_path = self.root_output_dir / 'prediction'
        self.prediction_path.mkdir(parents=True, exist_ok=True)

        self.prediction_csv_file_name = self.prediction_path / (cfg.NAME + '.csv')

    def get_logger(self):
        return self.logger
        
    def get_snapshot_path(self):
        return str(self.snapshot_path)

    def get_tb_path(self):
        return str(self.tensorboard_path)

    def get_prediction_path(self):
        return str(self.prediction_path)

    def get_prediction_csv_file_name(self):
        return str(self.prediction_csv_file_name)

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

class ModelWrapper:
    def __init__(self):
        pass
    def __call__(self, model, ema = False, wrapped_device = 'cpu'):
        if type(wrapped_device) == list and len(wrapped_device) > 1:
            model = nn.DataParallel(model, device_ids = wrapped_device)
        model.to(torch.device(wrapped_device if wrapped_device == 'cpu' else 'cuda:' + str(wrapped_device[0])))

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

'''
    make sure device without any spaces or tabs
    returned device is either [0, 1, ...] or cpu
'''
class DeviceWrapper:
    def __init__(self):
        pass
    def __call__(self, device):
        if 'cuda' in device:
            if ',' in device:
                device = list(map(int, device.split(':')[1].split(',')))
            else:
                device = [ int(device.split(':')[1]) ]
        return device

class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels):
    outputs = self.model(inputs)
    loss = self.loss(outputs, labels)
    # here convert to scalar to 1-d tensor for reduce operation
    return torch.unsqueeze(loss, 0), outputs

class ContrastiveFullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(ContrastiveFullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels):
    outputs, multiview_contrastive = self.model(inputs)
    loss = self.loss(outputs, labels)

    # here convert to scalar to 1-d tensor for reduce operation
    return torch.unsqueeze(loss, dim = 0), torch.unsqueeze(multiview_contrastive, dim = 0), outputs