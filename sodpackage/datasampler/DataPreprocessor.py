from .TrainRGBDDataset import TrainRGBDDataset
from .TestRGBDDataset import TestRGBDDataset
from .ValRGBDDataset import ValRGBDDataset
from .TrainPreTrainingDataset import TrainPreTrainingDataset
from .ValPreTrainingDataset import ValPreTrainingDataset
from .TestPreTrainingDataset import TestPreTrainingDataset

from torch.utils.data import DataLoader
import numpy as np

import random

def _make_loader(dataset, shuffle=True, drop_last=False):
    return DataLoaderX(dataset=dataset,
                       batch_size=arg_config["batch_size"],
                       num_workers=arg_config["num_workers"],
                       shuffle=shuffle, drop_last=drop_last,
                       pin_memory=True)

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        # get class
        TrainDataset = eval(self.config.TRAIN.DATASET)
        ValDataset = eval(self.config.VAL.DATASET)
        TestDataset = eval(self.config.TEST.DATASET)

        # instantiate
        self.train_dataset = TrainDataset(self.config.TRAIN.DATASET_ROOT, self.config.TRAIN.TRAIN_SIZE)
        self.val_dataset = ValDataset(self.config.VAL.DATASET_ROOT, self.config.TRAIN.TRAIN_SIZE)

        self.train_dataloader = DataLoader(self.train_dataset,
                                            batch_size = self.config.TRAIN.BATCH_SIZE,
                                            num_workers = self.config.TRAIN.WORKERS,
                                            pin_memory = True,
                                            shuffle = self.config.TRAIN.SHUFFLE,
                                            drop_last = True,
                                            worker_init_fn = lambda wid: random.seed(self.config.SEED + wid))
        # without shuffle and drop last
        self.val_dataloader = DataLoader(self.val_dataset,
                                        batch_size = self.config.TRAIN.BATCH_SIZE,
                                        num_workers = self.config.TRAIN.WORKERS,
                                        pin_memory = True,
                                        worker_init_fn = lambda wid: random.seed(self.config.SEED + wid))
        self.test_dataloaders = { }

        # list of 1-pair dictionaries
        for entry in self.config.TEST.DATASET_ROOTS:
            dataset_key, dataset_root = list(entry.items())[0]
            dataset = TestDataset(dataset_root, self.config.TRAIN.TRAIN_SIZE)
            # without shuffle and drop last
            dataloader = DataLoader(dataset,
                                    batch_size = self.config.TEST.BATCH_SIZE,
                                    num_workers = self.config.TEST.WORKERS,
                                    pin_memory = True,
                                    worker_init_fn = lambda wid: random.seed(self.config.SEED + wid))
            self.test_dataloaders[dataset_key] = dataloader

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_test_dataloaders(self):
        return self.test_dataloaders

    def get_val_dataloader(self):
        return self.val_dataloader