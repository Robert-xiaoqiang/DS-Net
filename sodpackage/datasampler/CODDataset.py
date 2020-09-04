import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image

import os

from .transforms.JointTransformer import JointCompose, JointResize, JointRandomHorizontallyFlip, JointRandomRotate

class CODDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, train_size):
        self.data_list = self._make_list(dataset_root, [ 'Image', 'GT' ])

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.joint_transform = JointCompose([
            JointResize(train_size),
            JointRandomHorizontallyFlip(),
            JointRandomRotate(10)
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            ])
        self.mask_transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, mask_path = self.data_list[index]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image, mask = self.joint_transform(image, mask)
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        image_main_name = os.path.splitext(os.path.basename(image_path))[0]

        return image, mask, image_main_name

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
            construct_print(f"数据文件夹中包含多种扩展名，这里仅使用{ext}")
        elif len(ext_list) == 1:
            ext = ext_list[0]
        else:
            raise NotImplementedError
        return main_list, ext

    def _make_list(self, root, keys):
        exts = [ ]
        main_file_names = set()
        for key in keys:
            keypath = os.path.join(root, key)
            cur_main_file_names, cur_ext = self._get_ext(os.listdir(keypath))
            main_file_names.update(cur_main_file_names)
            exts.append(cur_ext)    

        return [[ os.path.join(root, keys[i], main_file_name + exts[i]) 
                for i in range(len(keys)) ]
                for main_file_name in main_file_names]

