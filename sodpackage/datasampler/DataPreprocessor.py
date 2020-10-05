from .TrainRGBDDataset import TrainRGBDDataset
from .TestRGBDDataset import TestRGBDDataset
from .ValRGBDDataset import ValRGBDDataset
from .TrainPreTrainingDataset import TrainPreTrainingDataset
from .ValPreTrainingDataset import ValPreTrainingDataset
from .TestPreTrainingDataset import TestPreTrainingDataset

from torch.utils.data import DataLoader
import numpy as np

import random

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())

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

        self.train_dataloader = DataLoaderX(self.train_dataset,
                                            batch_size = self.config.TRAIN.BATCH_SIZE,
                                            num_workers = self.config.TRAIN.WORKERS,
                                            pin_memory = True,
                                            shuffle = self.config.TRAIN.SHUFFLE,
                                            drop_last = True,
                                            worker_init_fn = lambda wid: random.seed(self.config.SEED + wid))
        # without shuffle and drop last
        self.val_dataloader = DataLoaderX(self.val_dataset,
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
            dataloader = DataLoaderX(dataset,
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