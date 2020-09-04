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
import csv

import os
import copy
from collections import OrderedDict

from ..helper.TrainHelper import AverageMeter, LoggerPather, DeviceWrapper
from ..helper.TestHelper import Evaluator, FullModelForTest

class Deducer:
    def __init__(self, model, test_dataloaders, config):
        self.model = model
        self.test_dataloaders = test_dataloaders
        self.config = config

        cudnn.benchmark = self.config.CUDNN.BENCHMARK
        cudnn.deterministic = self.config.CUDNN.DETERMINISTIC
        cudnn.enabled = self.config.CUDNN.ENABLED

        self.wrapped_device = DeviceWrapper()(config.DEVICE)
        self.main_device = torch.device(self.wrapped_device if self.wrapped_device == 'cpu' else 'cuda:' + str(self.wrapped_device[0]))

        self.model = FullModelForTest(self.model)

        self.model.to(self.main_device)
        if type(self.wrapped_device) == list:
            self.model = nn.DataParallel(self.model, device_ids = self.wrapped_device)

        loggerpather = LoggerPather(self.config)
        self.logger = loggerpather.get_logger()
        self.snapshot_path = loggerpather.get_snapshot_path()
        self.prediction_path = loggerpather.get_prediction_path()
        self.csv_file_name = loggerpather.get_prediction_csv_file_name()

        self.loaded_epoch = None
        self.to_pil = transforms.ToPILImage()

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
            self.loaded_epoch = params['epoch']
            return True
    
    def build_data(self, batch_data):
        batch_data = [ d.to(self.main_device, non_blocking=True) if torch.is_tensor(d) else d for d in batch_data ]
        return tuple(batch_data)

    def build_test_model(self):        
        b = self.load_checkpoint(snapshot_key = 'best')
        if b:
            self.logger.info('loaded successfully, test based on model from best epoch {}'.format(self.loaded_epoch))
        else:
            self.logger.info('loaded failed, test based on ImageNet scratch')

    def deduce(self):
        self.build_test_model()
        self.model.eval()

        csv_head = [ 'dataset_key', 'S', 'MAXE', 'MAXF', 'MAE' ]
        # csv_head = [ 'dataset_key', 'S', 'MAE' ]
        # csv_head = [ 'dataset_key', 'WeightedF' ]
        csv_stuff = [ ]

        for dataset_key, dataloader in self.test_dataloaders.items():
            self.logger.info('Test on {}'.format(dataset_key))
            save_path = os.path.join(self.prediction_path, dataset_key)
            os.makedirs(save_path, exist_ok = True)

            preds = [ ]
            masks = [ ]
            tqdm_iter = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for batch_id, batch_data in tqdm_iter:
                tqdm_iter.set_description(f'Infering: te=>{batch_id + 1}')
                with torch.no_grad():
                    batch_rgb, batch_label, batch_mask_path, batch_key, \
                    = self.build_data(batch_data)
                    output = self.model(batch_rgb)

                output_cpu = output[0].cpu().detach()
                for pred, mask_path, image_main_name in zip(output_cpu, batch_mask_path, batch_key):
                    mask = copy.deepcopy(Image.open(mask_path).convert('L'))
                    
                    pred = self.to_pil(pred).convert('L').resize(mask.size)
                    pred.save(os.path.join(save_path, image_main_name + '.png'))
                    
                    preds.append(pred)
                    masks.append(mask)
            self.logger.info('Start evaluation for dataset {}'.format(dataset_key))
            results = Evaluator.evaluate(preds, masks)
            self.logger.info('Finish evaluation for dataset {}'.format(dataset_key))
            results['dataset_key'] = dataset_key
            csv_stuff.append(results)

        self.logger.info('Finish testing, let\'s, save results in {}'.format(self.csv_file_name))
        with open(self.csv_file_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames = csv_head)
            writer.writeheader()
            for row in csv_stuff:
                writer.writerow(row)
        self.logger.info('Finish saving it, enjoy everything')

    # predict only without evaluation
    def predict(self):
        self.build_test_model()
        self.model.eval()

        for dataset_key, dataloader in self.test_dataloaders.items():
            self.logger.info('Test on {}'.format(dataset_key))
            save_path = os.path.join(self.prediction_path, dataset_key)
            os.makedirs(save_path, exist_ok = True)

            tqdm_iter = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for batch_id, batch_data in tqdm_iter:
                tqdm_iter.set_description(f'Infering: te=>{batch_id + 1}')
                with torch.no_grad():
                    batch_rgb, batch_label, batch_mask_path, batch_key, \
                    = self.build_data(batch_data)
                    output = self.model(batch_rgb)

                output_cpu = output[0].cpu().detach()
                for pred, mask_path, image_main_name in zip(output_cpu, batch_mask_path, batch_key):
                    mask = copy.deepcopy(Image.open(mask_path).convert('L'))
                    
                    pred = self.to_pil(pred).convert('L').resize(mask.size)
                    pred.save(os.path.join(save_path, image_main_name + '.png'))

        self.logger.info('Finish predicting it, enjoy everything')