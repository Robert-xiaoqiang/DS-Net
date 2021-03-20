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

class RGBDDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, train_size):
        super().__init__()
        self.data_list = self._make_list(dataset_root, [ 'RGB', 'depth', 'GT' ])

        # rebuttal 3:7 setting, create a new class TrSplitRGBDDataset is better
        self.data_list = self.data_list[:int(0.7 * len(self.data_list))]

        self.mean = np.array([0.447, 0.407, 0.386])
        self.std = np.array([0.244, 0.250, 0.253])

        self.joint_transform = JointCompose([
            JointResize(train_size),
            JointRandomHorizontallyFlip(),
            JointRandomRotate(10)
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            ])
        self.depth_transform = transforms.ToTensor()
        self.mask_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        raise NotImplementedError

    def _get_ext(self, path_list):
        main_list = set()
        ext_list = set()
        for p in path_list:
            main_file_name, extension_name = os.path.splitext(p)
            main_list.add(main_file_name)
            ext_list.add(extension_name)
        ext_list = list(ext_list)
        if len(ext_list) != 1:
            if '.png' in ext_list:
                ext = '.png'
            elif '.jpg' in ext_list:
                ext = '.jpg'
            elif '.bmp' in ext_list:
                ext = '.bmp'
            else:
                raise NotImplementedError
            pprint('There are multiple kinds of extension in this directory, we just use `{}`'.format(ext))
        elif len(ext_list) == 1:
            ext = ext_list[0]
        else:
            raise NotImplementedError
        return main_list, ext

    def _make_list(self, root, keys):
        main_file_names = None
        exts = [ ]
        for key in keys:
            keypath = os.path.join(root, key)
            cur_main_file_names, cur_ext = self._get_ext(os.listdir(keypath))
            # main_file_names.update(cur_main_file_names) |=
            main_file_names = cur_main_file_names if main_file_names is None else main_file_names & cur_main_file_names
            exts.append(cur_ext)

        return [[ os.path.join(root, keys[i], main_file_name + exts[i]) 
                for i in range(len(keys)) ]
                for main_file_name in main_file_names]
