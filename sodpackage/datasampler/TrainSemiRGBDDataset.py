import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from pprint import pprint

import os

from .transforms.JointTransformer import JointCompose, JointResize, JointRandomHorizontallyFlip, JointRandomRotate
from .SemiRGBDDataset import SemiRGBDDataset

class TrainSemiRGBDDataset(SemiRGBDDataset):
    def __init__(self, labeled_root, unlabeled_root, train_size):
        super().__init__(labeled_root, unlabeled_root, train_size)
    
    def __getitem__(self, index):
        if index < len(self.labeled_list):
            image_path, depth_path, mask_path = self.labeled_list[index]
            image = Image.open(image_path).convert('RGB')
            depth = Image.open(depth_path).convert('L')
            mask = Image.open(mask_path).convert('L')

            image, depth, mask = self.joint_transform(image, depth, mask)
            image = self.labeled_image_transform(image)
            depth = self.depth_transform(depth)
            mask = self.mask_transform(mask)

            return image, depth, mask
        else:
            index -= len(self.labeled_list)
            image_path, depth_path = self.unlabeled_list[index]
            image = Image.open(image_path).convert('RGB')
            depth = Image.open(depth_path).convert('L')

            image, depth = self.joint_transform(image, depth)
            image = self.unlabeled_image_transform(image)
            depth = self.depth_transform(depth)
            # here cpu float tensor, dataloader is just on cpu before being feeded into model
            mask = torch.zeros_like(depth)

            return image, depth, mask