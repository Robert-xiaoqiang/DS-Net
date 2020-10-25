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
MTDisentangleFullModel, MTFakeFullModel
from ..helper.TestHelper import Evaluator
from ..inference.Deducer import Deducer

from .MTSemiSupervisedTrainer import MTSemiSupervisedTrainer

class MTDisentangleSemiSupervisedTrainer(MTSemiSupervisedTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloaders, config):
        # polymorphism in constructor, let's build some attributes before it !!!!
        self.ema_model = copy.deepcopy(model)
        super().__init__(model, train_dataloader, val_dataloader, test_dataloaders, config)

    def wrap_model(self):
        self.model = MTDisentangleFullModel(self.model, self.criterion)
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
