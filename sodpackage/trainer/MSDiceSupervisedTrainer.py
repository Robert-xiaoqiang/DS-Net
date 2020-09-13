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
MSDiceFullModel, DiceLoss
from ..helper.TestHelper import Evaluator
from ..inference.Deducer import Deducer

from .MSSupervisedTrainer import MSSupervisedTrainer

class MSDiceSupervisedTrainer(MSSupervisedTrainer):
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
