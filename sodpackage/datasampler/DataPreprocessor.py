from .TrainCODDataset import TrainCODDataset
from .TestCODDataset import TestCODDataset

from torch.utils.data import DataLoader
import numpy as np

import random

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.train_dataset = TrainCODDataset(self.config.TRAIN.DATASET_ROOT, self.config.TRAIN.TRAIN_SIZE)
        self.val_dataset = TestCODDataset(self.config.VAL.DATASET_ROOT, self.config.TRAIN.TRAIN_SIZE)

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

        # list of 1-pair dictionary
        for entry in self.config.TEST.DATASET_ROOTS:
            dataset_key, dataset_root = list(entry.items())[0]
            dataset = TestCODDataset(dataset_root, self.config.TRAIN.TRAIN_SIZE)
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